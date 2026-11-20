import sys, os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from directory_config import *
from dataproc import *
from prompts import construct_8values_prompt

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" #ignored?
os.environ["HF_HOME"] = cache_base_directory + "hf_cache"  # Set cache directory for Hugging Face models
os.environ["HF_MODEL_CACHE"] = cache_base_directory + "hf_cache"
os.environ["HF_DATASETS_CACHE"] = cache_base_directory + "hf_cache"

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, Mistral3ForConditionalGeneration
import torch, json, gc, numpy as np, pickle as pkl, argparse
from scipy.spatial.distance import cosine
torch.manual_seed(42)
from repeng_scripts.repeng import ControlModel, ControlVector, DatasetEntry, RFMToolkit
from repeng_scripts.repeng.extract import batched_get_hiddens
from scipy.stats import pearsonr, ttest_ind
from itertools import product


def chat_template_unparse(messages: list[dict]) -> str:
    template = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        template.append(
            f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        )
    if messages[-1]["role"] != "assistant":
        # prefill assistant prefix
        template.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return " ".join(template)
    else:
        template = " ".join(template)
        return template[:-10] #cut off eot token to allow generation

def chat_template_parse(resp: str) -> list[tuple[str, str]]:
    resp = resp.strip().removeprefix("<|begin_of_text|>")
    messages = []
    for part in resp.split("<|start_header_id|>"):
        role_and_content = part.split("<|end_header_id|>")
        if len(role_and_content) == 1:
            role, content = role_and_content[0], ""
        else:
            role, content = role_and_content
        content = content.split("<|eot_id|>")[0]
        messages.append((role.strip(), content.strip()))
    return messages

def make_dataset(
    template: str,
    positive_personas: list[list[str]],
    negative_personas: list[list[str]],
    suffix_list: list[str],
) -> list[DatasetEntry]:
    dataset = []
    for suffix in suffix_list:
        for positive_persona, negative_persona in zip(
            positive_personas, negative_personas
        ):
            positive_template = template.format(*positive_persona)
            negative_template = template.format(*negative_persona)
            dataset.append(
                DatasetEntry(
                    positive=f"{positive_template}{suffix}",
                    negative=f"{negative_template}{suffix}",
                )
            )
    return dataset

def löwdin_orthogonalization(vectors:list, eps: float = 1e-9) -> list[torch.Tensor]:
    """
    Performs Löwdin (Symmetric) Orthogonalization on a set of vectors and normalizes. Man be equivalent to Singular Value Decomposition.
    
    Parameters:
    - vectors: a list of 1D torch.Tensors od same dimension
    - eps: A small epsilon to avoid division by zero for near-zero eigenvalues.
         
    Returns:
    - list of torch.Tensor containing the orthogonalized vectors.
    """
    A = torch.stack(vectors, dim=1)
    # 1. Compute the overlap matrix S = A^T * A
    # S will be of shape (N, N)
    S = torch.mm(A.t(), A)
    
    # 2. Eigen-decomposition of S
    # L contains eigenvalues, U contains eigenvectors
    L, U = torch.linalg.eigh(S)
    
    # 3. Compute L^(-1/2) safely
    # We clamp to avoid issues with floating-point noise or near-zero eigenvalues
    L_clamped = torch.clamp(L, min=eps)
    L_inv_sqrt = 1.0 / torch.sqrt(L_clamped)
    
    # 4. Reconstruct S^(-1/2) = U * diag(L^(-1/2)) * U^T
    # Using entry-wise multiplication with U for efficiency
    S_inv_sqrt = torch.mm(U * L_inv_sqrt, U.t())
    
    # 5. Compute the symmetric orthogonalized vectors Q = A * S^(-1/2)
    Q = torch.mm(A, S_inv_sqrt)

    Q_normalized = torch.nn.functional.normalize(Q, p=2, dim=0)    
    
    return list(torch.unbind(Q_normalized, dim=1))

def subspace_projection(vector_list):
    """
    Disentangles a list of concept vectors by removing components 
    that are linear combinations of the other vectors.
    
    Parameters:
    vector_list (list of torch.Tensor): List of 1D tensors of the same shape.
    
    Returns:
    list of torch.Tensor: The uniquely disentangled concept vectors.
    """
    # 1. Stack vectors into a matrix V of shape (dimensions, num_concepts)
    V = torch.stack(vector_list, dim=1).float()
    num_concepts = V.shape[1]
    disentangled_matrix = torch.zeros_like(V)
    
    for i in range(num_concepts):
        # Isolate the target vector v_i
        v_i = V[:, i]
        
        # Get all other vectors: shape (dimensions, num_concepts - 1)
        indices = [j for j in range(num_concepts) if j != i]
        X_minus_i = V[:, indices]
        
        # 2. Compute the projection matrix onto the subspace of the other vectors
        # P = X * pinv(X)
        # pinv is used instead of inverse for numerical stability against collinearity
        X_pinv = torch.linalg.pinv(X_minus_i)
        P = X_minus_i @ X_pinv
        
        # 3. Subtract the projection from the original vector
        # residual = v_i - (P @ v_i)
        disentangled_matrix[:, i] = v_i - torch.mv(P, v_i)
        disentangled_matrix = torch.nn.functional.normalize(disentangled_matrix, p=2, dim=0)
        
    # 4. Unbind back into a list of individual 1D tensors
    return list(torch.unbind(disentangled_matrix, dim=1))

def orthogonalize_per_layer(vectors:list[ControlVector], method="subspace") -> list[ControlVector]:
    """
    Applies selected orthogonalization technique and returns orthogonalized vectors
    """
    for layer in vectors[0].directions:
        l_vecs = [v.directions[layer] for v in vectors]
        if method == "löwdin":
            l_vecs_orth = löwdin_orthogonalization(l_vecs)
        elif method == "subspace":
            l_vecs_orth = subspace_projection(l_vecs)
        else:
            raise Exception("unknown orthogonalization")
        for i, v in enumerate(vectors):
            v.directions[layer] = l_vecs_orth[i]
    return vectors

class Model_with_steering:
    def __init__(self, model_type, dataset=None, steering_str="", layer_ids=None, strength=1, temperature=0,  method="rfm"):
        self.steering_str=steering_str
        self.model_type = model_type
        gc.collect()
        torch.cuda.empty_cache()
        self.strength = strength
        self.temperature = temperature
        if not dataset:
            print("Warning: No steering specified. Please make sure to train vector before running inference.")

        if model_type == "llama":
            self.model_name = "meta-llama/Llama-3.1-8B-Instruct"
        elif model_type == "llama-big":
            self.model_name = "meta-llama/Llama-3.3-70B-Instruct"
        elif model_type == "qwen":
            self.model_name = "Qwen/Qwen3.5-9B"    
        elif model_type == "mistral-medium":
            self.model_name = "mistralai/Mistral-Medium-3.5-128B"    
        else:
            raise Exception(f"Model type {model_type} not yet implemented for representation steering model")

        ds_name = dataset[0]["name"] if type(dataset) == list else dataset
        self.model_string = self.model_name.split("/")[-1] + f"_strength_{self.strength}_steered_{ds_name}"

        if model_type == "mistral-medium":
            self.tokenizer = AutoProcessor.from_pretrained(self.model_name).tokenizer
            model = Mistral3ForConditionalGeneration.from_pretrained(self.model_name, device_map="auto")
            model.config.num_hidden_layers = 88
        else:
            model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token_id = 0

        #Llama 8B has 32 hidden layers, 70B has 80 (of which the example steers #20-60)
        # For Mistral-7B, which also has 32 hidden layers, they steer ~#15-27
        # Beaglehole steer list(range(-8, -31, -1))
        if layer_ids:
            self.layer_ids = list(layer_ids)
        else:
            try:
                num_hidden_layers = model.config.num_hidden_layers
                layer_ids = list(range(10+int(num_hidden_layers*0.125), int(num_hidden_layers))) #TODO changed: steering until last layer, not *0.85
            except:
                layer_ids = list(range(-5, -18, -1))
        print("steering layers", layer_ids)
        wrapped_model = model

        self.model = ControlModel(wrapped_model, model_type, layer_ids=layer_ids) 
        self.vectors = []
        if dataset:
            self.add_vector(dataset=dataset, method=method)

    def get_steering_vectors(self, datasets:list[dict], ds_template="{}", use_default_suffixes=True, method="rfm"):
        vectors = []
        if use_default_suffixes:
            with open(this_base_directory + "RepresentationPoliticalLLM/repeng_scripts/notebooks/data/all_truncated_outputs.json") as f:
                output_suffixes = json.load(f)
                try:
                    suffixes = [
                    self.tokenizer.convert_tokens_to_string(tokens[:i])
                    for tokens in (self.tokenizer.tokenize(s) for s in output_suffixes)
                    for i in range(1, len(tokens))]
                except:
                    suffixes = []
        else:
            suffixes = [""]
        for dataset in datasets:
            chat_template  =[{"role": "assistant", "content": ds_template}]
            if dataset["positive"] and len(dataset["positive"][0]) == 2:
                chat_template.insert(0, {"role": "user", "content": "{}"})
            if method == "pca":
                ds = make_dataset(chat_template_unparse(chat_template), dataset["positive"], dataset["negative"], suffixes)
                vectors.append(ControlVector.train(self.model, self.tokenizer, dataset=ds, batch_size=32, method="pca_center", name=dataset["name"] if "name" in dataset else "name"))
            elif method == "rfm":
                toolkit = RFMToolkit()
                train_data = dataset["positive"] + dataset["negative"]
                train_data = [self.tokenizer.apply_chat_template([{"role": ["user", "assistant"][i], "content": prompt[i]} for i in range(len(prompt))], tokenize=False, continue_final_message=True) for prompt in train_data]
                #print("debug: ", train_data[0])
                train_labels = [1] * len(dataset["positive"]) + [0] * len(dataset["negative"])
                combined = list(zip(train_data, train_labels))
                random.Random(15).shuffle(combined)
                data, labels = zip(*combined)
                train_data = list(data)
                train_labels = list(labels)
                print("train data size: ", len(train_labels))
                directions, signs, detector_coefs, _ = toolkit._compute_directions(train_data, 
                                                                train_labels, 
                                                                val_data=None, val_labels=None,
                                                                model=self.model, tokenizer=self.tokenizer, 
                                                                hidden_layers=list(range(-1, -self.model.config.num_hidden_layers, -1)), 
                                                                hyperparams={'control_method' : "rfm", 'rfm_iters' : 10, 'forward_batch_size' : 2, 'M_batch_size' : 2048, 'n_components' : 1},
                                                            )
                vector = ControlVector(model_type=self.model_type, directions={self.model.config.num_hidden_layers+layer: components[0] for layer, components in directions.items()}, name=dataset["name"] if "name" in dataset else "name")
                vectors.append(vector)
            print(f"Vector trained: {dataset['name']if 'name' in dataset else 'unnamed'}")
        self.vectors = vectors
        return vectors

    def add_vector(self, dataset:str|list, method="rfm"):
        new_vec = None
        ds_name = dataset[0]["name"] if type(dataset) == list else dataset

        pkl_path = this_base_directory + "RepresentationPoliticalLLM/steering_vectors/" + f"{self.model_type}_{ds_name}_{method}.pkl"
        if os.path.exists(pkl_path):
            new_vec = pkl.load(open(pkl_path, "rb"))
            print(f"Loaded steering vectors for dataset {ds_name} from pkl file")
        elif type(dataset) == str:
            dataset = get_political_dataset(dataset=dataset)
            if not dataset:
                raise Exception(f"Unknown dataset passed to steering model: {ds_name} or other error during dataset generation")
        elif type(dataset) != list:
            print("Warning: add_vector failed due to invalid dataset input. No vector was added")
            return
   
        vectors_old = self.vectors
        if new_vec is None:
            #use_default_suffixes = len(ds[0]["positive"]) < 30
            new_vec = self.get_steering_vectors(dataset, method=method)
            print(f"Vector {ds_name} trained and set as model vector")
            pkl.dump(new_vec, open(pkl_path, "wb"))
        self.vectors = vectors_old + new_vec
        self.vector = new_vec

    
    def map_to_vectors(self, prompt:str, measure="cosine"):
        """collect activations for prompt, then calculate projection along each of model.vectors (counting only layers in self.model.layer_ids)
        Returns dict {axis_name: score}
        """
        input = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        activations = batched_get_hiddens(self.model, self.tokenizer, inputs=[input], hidden_layers=self.model.layer_ids, batch_size=1) #dictionary layer_id -> (n_samples (1) x hidden_dim)
        alignments = {}
        for v in self.vectors:
            if measure == "cosine":
                scores = [1-cosine(activations[i][0], v.directions[i].cpu().numpy()).item() for i in self.model.layer_ids] #we want similarity (-1 to 1), which is 1-distance
            elif measure == "dot":
                scores = [np.dot(activations[i][0], v.directions[i].cpu().numpy()).item() for i in self.model.layer_ids]
            else:
                raise Exception(f"Invalid measure {measure}. Options are 'cosine' and 'dot'")
            alignments[v.name] = sum(scores)/len(scores)
        return alignments
    
    def process_vector_mapping(self, items:list[str], prompt_template:str="{}", axis:str|None=None):
        """
        Output format: List of dicts {axis: score} for each item in items. Item names are not passed.
        """
        all_scores = []
        for item in items:
            scores = self.map_to_vectors(prompt_template.format(item))
            if axis is not None:
                scores = scores[axis]
            all_scores.append(scores)
        #print("debug", all_scores)
        return all_scores

    def prompt_model(self, prompt:list|str, max_tokens=10000, steering=False) -> str:
        """
        Able to handle prompting with strings and chat templates
        """
        if type(prompt) == list: #already in chat template form
                messages = prompt
        else:
            messages = [{"role": "user", "content": prompt}]
        continue_final_message = messages[-1]["role"] == 'assistant'
        input = self.tokenizer.apply_chat_template(messages, tokenize=True, enable_thinking=False, return_tensors="pt",
                                                   add_generation_prompt=not continue_final_message, continue_final_message=continue_final_message)
        return self.generate_with_vector(input=input, 
                             vector=self.vector, strength=self.strength, max_new_tokens=max_tokens,
                             temperature=self.temperature)

    def generate_with_vector(self, input: str, vector: ControlVector, strength = 1, max_new_tokens: int = 128,
                             repetition_penalty: float = 1.1, temperature: float = 0.7,) -> str:

        input_ids = input.to("cuda:0")
        settings = {
            "pad_token_id": self.tokenizer.eos_token_id,  # silence warning
            "do_sample": temperature > 0,   # we want do_sample iff temperature is set to above 0
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            #"return_full_text" : False
        }
        if vector:
            self.model.set_control(vector, strength)
        response = self.model.generate(**input_ids, **settings) 
        response = response[0][input_ids["input_ids"].shape[1]:]
        response = self.tokenizer.decode(response, skip_special_tokens=True).strip()
        if response.startswith("assistant"):
            response = response[9:].strip()
        self.model.reset()
        return response

from scipy.spatial.distance import cosine
from itertools import combinations
import copy

def avg_angle(v1: ControlVector, v2: ControlVector):
    assert v1.directions.keys() == v2.directions.keys()
    dists = []
    for i in v1.directions.keys():
        for vec in [v1, v2]:
            if type(vec.directions[i]) == torch.Tensor: vec.directions[i] = vec.directions[i].cpu()
        dists.append(cosine(v1.directions[i], v2.directions[i]))
    #print(dists)
    return sum(dists)/len(dists)


def vec_sum(vectors:list, signs=[]):
    if signs:
        assert len(signs) == len(vectors)
    else:
        signs = [1] * len(vectors)
    res = copy.deepcopy(vectors[0])
    for layer in res.directions.values():
        layer *= signs[0]
    for i, vec in enumerate(vectors[1:]):
        for layer in res.directions.keys():
            res.directions[layer] = res.directions[layer] + signs[i+1] * vec.directions[layer]
    if any(signs): #we can't normalize a zero vector
        for layer in res.directions.keys():
            res.directions[layer] *= (1/np.linalg.norm(res.directions[layer].cpu())) #normalize
            assert abs(np.linalg.norm(res.directions[layer].cpu()) - 1) < .001

    return res


def get_all_angles(model_type, method="pca"):
    statements = get_politician_statements(return_list=True)
    all_left_statements = [s[1] for s in statements if s[0]<0]
    most_left_statements = all_left_statements[:100]
    mod_left_statements = all_left_statements[-100:]
    all_right_statements = [s[1] for s in statements if s[0]>0]
    mod_right_statements = all_right_statements[:100]
    most_right_statements = all_right_statements[-100:]
    #socialist_statements = get_far_left_statements()

    model = Model_with_steering(model_type)

    vectors = {"left-modleft": (most_left_statements, mod_left_statements),
               "modleft-modright": (mod_left_statements, mod_right_statements),
            "modright-right": (mod_right_statements, most_right_statements),
            #"left-modright": (most_left_statements, mod_right_statements),
            #"left-right": (most_left_statements, most_right_statements),
            #"modleft-right": (mod_left_statements, most_right_statements),
            "main": (all_left_statements, all_right_statements),
            }
    vectors = [{"name": name, "positive": pos, "negative": neg} for name, (pos, neg) in vectors.items()]
    st_vectors = model.get_steering_vectors(vectors, use_default_suffixes=False, method=method)
    # for i in range(len(st_vectors)):
    #     print(f"{vectors[v1]['name']}: avg length {avg_len(st_vectors[i])}")
    # del model
    # return st_vectors
    for v1, v2 in combinations(range(len(vectors)), 2):
        print(f"avg angle between {vectors[v1]['name']} and {vectors[v2]['name']}: {avg_angle(st_vectors[v1], st_vectors[v2])}")

def signal_by_contrastive_data(model_type):
    statements = get_politician_statements(return_list=True)
    all_left_statements = [s[1] for s in statements if s[0]<0]
    all_right_statements = [s[1] for s in statements if s[0]>0]

    model = Model_with_steering(model_type, "None", "")
    random_left = random.Random(15).sample(all_left_statements, 20)
    random_right = random.Random(15).sample(all_right_statements, 20)
    lr_vectors = [{"name": i, "positive": [pos], "negative": [neg]} for i, pos, neg in zip(range(len(random_left)), random_right, random_left)]
    ll_vectors = [{"name": i, "positive": [pos], "negative": [neg]} for i, (pos, neg) in [(i, random.Random(i).sample(random_left, 2)) for i in range(20)]]
    rr_vectors = [{"name": i, "positive": [pos], "negative": [neg]} for i, (pos, neg) in [(i, random.Random(i).sample(random_right, 2)) for i in range(20)]]
    for vec, name in [(ll_vectors, "ll"), (lr_vectors, "lr"),  (rr_vectors, "rr")]:
        vectors_st = model.get_steering_vectors(vec, use_default_suffixes=True, method="pca")
        avg_angles = [avg_angle(vectors_st[v1], vectors_st[v2]) for v1, v2 in combinations(range(len(vec)), 2)]
        print(f"{name}: avg avg angle between pairs of vectors: {sum(avg_angles)/len(avg_angles)}, Variance of angles: {sum([(a - sum(avg_angles)/len(avg_angles))**2 for a in avg_angles])/len(avg_angles)}")
        #avg_lens = [avg_len(v) for v in vectors_st]
        #print(f"{name}: avg avg length: {sum(avg_lens)/len(avg_lens)}")
 
def signal_by_dataset_size(model_type):
    statements = get_politician_statements(return_list=True)
    all_left_statements = [s[1] for s in statements if s[0]<0]
    all_right_statements = [s[1] for s in statements if s[0]>0]

    model = Model_with_steering(model_type, "None", "")
    # we run different dataset sizes of contrastive pairs, we always compare across 50 vectors
    # naive seeding would cause each smaller size to be contained in the bigger, we avoid that
    poss_combos = len(all_left_statements) * len(all_right_statements)
    print(len(all_left_statements), len(all_right_statements))
    for size in [1, 5, 10, 50, 100, 200]:
        print(f"Possible pairs: {poss_combos}. Expected overlap of pairs in the training set of 2 vectors is {(size/poss_combos) * (size/poss_combos)}")
        lr_vectors = [{"name": i, "positive": pos, "negative": neg} for i, pos, neg in [(i, random.Random(100*size+i).sample(all_right_statements, size), random.Random(100*size+i+.1).sample(all_left_statements, size)) for i in range(50)]]
        vectors_st = model.get_steering_vectors(lr_vectors, use_default_suffixes=False)
        avg_angles = [avg_angle(vectors_st[v1], vectors_st[v2]) for v1, v2 in combinations(range(len(lr_vectors)), 2)]
        print(f"{size}: avg avg angle between pairs of vectors: {sum(avg_angles)/len(avg_angles)}, Variance of angles: {sum([(a - sum(avg_angles)/len(avg_angles))**2 for a in avg_angles])/len(avg_angles)}")


def vectors_different(datasets):
    model = Model_with_steering("llama", "None", "")
    for candidate_ds in range(len(datasets)):
        for i, statement in enumerate(datasets[candidate_ds]["positive"]):
            datasets.append({"name": f"{datasets[candidate_ds]['name']}-{i}", "positive": [statement], "negative": [datasets[candidate_ds]["negative"][i]]})
    st_vectors = model.get_steering_vectors(datasets, use_default_suffixes=False)
    results = {"nn": [], "nr": [], "rr": []}
    for v1, v2 in combinations(range(len(datasets)), 2):
        avg_ang = avg_angle(st_vectors[v1], st_vectors[v2])
        #print(f"avg angle between {datasets[v1]['name']} and {datasets[v2]['name']}: {avg_ang}")
        if "-" in datasets[v1]['name'] and "-" in datasets[v2]['name']:
            results["".join(sorted([datasets[v1]['name'][0], datasets[v2]['name'][0]]))].append(avg_ang)
    for k, v in results.items():
        print(k, sum(v)/len(v))
    print(ttest_ind(results["nr"], results["nn"] + results["rr"]))

def load_refusals(prefixes=[""]):
    responses = {"neutral": [], "refusal": []}
    with open(os.path.join(this_base_directory + "RepresentationPoliticalLLM/refusals.csv")) as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            for prompt in prefixes:
                responses[row[2]].append([prompt, row[0]])
    return responses["neutral"], responses["refusal"]

def evaluate_scaling(scores:dict[str, dict[str,float]], manifesto_scores:dict[str, dict[str,float]], baseline="axes_combined"):
    """
    For each axis, calculates correlation between axis scores and manifesto scores
    both should be {"party": {"axis1": 2, "axis2": ...}}
    """
    manifesto_to_topic = {"per104": "Military-pro", "per107": "Internationalism-pro", "per108": "EU-pro", "per201": "Freedoms", "per501": "Environment", "per503": "Equality"}
    axis_scores = {axis: [] for axis in list(scores.values())[0] if axis in ["econ", "govt", "scty", "dipl", baseline]}
    manifesto_axis_scores = {axis: [] for axis in list(manifesto_scores.values())[0]}
    for party, p_scores in scores.items():
        if type(p_scores) == dict and party in manifesto_scores:
            for axis in axis_scores: axis_scores[axis].append(p_scores[axis])
            for axis in manifesto_axis_scores: 
                if manifesto_scores[party][axis] != "NA":
                    manifesto_axis_scores[axis].append(manifesto_scores[party][axis])
    correlations = {}
    pvalues = {}
    for axis, manifesto_axis in product(axis_scores.keys(), manifesto_axis_scores.keys()):
        if len(axis_scores[axis]) == len(manifesto_axis_scores[manifesto_axis]): # all parties present
            corr = pearsonr(axis_scores[axis], manifesto_axis_scores[manifesto_axis])
            issue = manifesto_to_topic[manifesto_axis] if manifesto_axis in manifesto_to_topic else manifesto_axis
            print(f"{axis}, {issue}: r = {round(corr.statistic,2)}, pvalue {round(corr.pvalue, 3)}")
            correlations[axis + "-" + manifesto_axis] = corr.statistic
            pvalues[axis + "-" + manifesto_axis] = corr.pvalue
    return correlations

def scaling_winrate(manifesto_scores, scores:dict[str, dict[str,float]], baseline=None):
    # scores should include baseline(s)
    axis_scores = {axis: [] for axis in list(scores.values())[0] if axis in ["econ", "govt", "scty", "dipl", baseline, "econ_pos","econ_neg","govt_pos","govt_neg", "scty_pos","scty_neg","dipl_pos","dipl_neg"]}
    manifesto_axis_scores = {axis: [] for axis in list(manifesto_scores.values())[0]}
    for party, p_scores in scores.items():
        if type(p_scores) == dict and party in manifesto_scores:
            for axis in axis_scores: axis_scores[axis].append(p_scores[axis])
            for axis in manifesto_axis_scores: 
                if manifesto_scores[party][axis] != "NA":
                    manifesto_axis_scores[axis].append(manifesto_scores[party][axis])
    # print("debug: ", manifesto_axis_scores)
    # print("debug: ", axis_scores)
    wins = {axis: [] for axis in axis_scores}
    num_significant = 0
    for manifesto_axis in manifesto_axis_scores:
        #assert len(scores) == len(manifesto_axis_scores[manifesto_axis]) # all parties present
        corrs = [pearsonr(list(axis_scores[axis]), list(manifesto_axis_scores[manifesto_axis])).statistic for axis in axis_scores]
        significances = [pearsonr(list(axis_scores[axis]), list(manifesto_axis_scores[manifesto_axis])).pvalue < 0.05 for axis in axis_scores]
        max_corr = list(axis_scores.keys())[numpy.argmax(corrs)]
        if significances[numpy.argmax(corrs)] < 0.05: num_significant += 1
        wins[max_corr].append(manifesto_axis)
    print(f"Of {len(manifesto_axis_scores)} variables, {num_significant} are significant, a ratio of {round(num_significant/len(manifesto_axis_scores), 3)}")
    winrates = {}
    for axis in wins:
        winrates[axis] = len(wins[axis])/len(manifesto_axis_scores)
    return winrates, wins
