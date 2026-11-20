import sys, os, json, re, argparse, pickle as pkl, numpy as np, copy, pingouin 
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from directory_config import *
from dataproc import *
from repr_steering import Model_with_steering, vec_sum, orthogonalize_per_layer, evaluate_scaling, scaling_winrate, get_all_angles, avg_angle
from prompts import *
from models import Model

from itertools import product, combinations
import matplotlib.pyplot as plt
from matplotlib import lines
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr, pearsonr, ttest_ind, ttest_rel


VALUES = {
    "econ": {"positive": "equality", "negative": "markets", "name": "economic"},
    "govt": {"positive": "liberty", "negative": "authority", "name": "civil"},
    "scty": {"positive": "progress", "negative": "tradition", "name": "societal"},
    "dipl": {"positive": "globe", "negative": "nation", "name": "diplomatic"},
        }

country_to_parties = {"Germany": ["AfD", "CDU/CSU", "FDP", "SPD", "90/Greens", "LINKE"],
                          "Italy": ["FDI", "League", "FI", "M5S", "PD"],
                          "France": ["Horizons", "Together for the Republic", "Democratic Movement", "National Front", "The Republicans", "Socialist Party", "New Ecologic and Social People's Union"],
                          "United Kingdom": ["GPEW", "Labour", "LibDems", "Reform UK", "Conservatives", "The Party of Wales", "SNP", "DUP"],
                          "Sweden": ["Centre Party", "Sweden Democrats", "Moderate Coalition Party", "Christian Democrats", "Liberals", "Social Democratic Labour Party", "Left Party", "Green Ecology Party"]}

country_to_adj = {"Germany": "German", "Italy": "Italian","France": "French","United Kingdom": "British", "Sweden": "Swedish"}

def get_axes_vectors(datasets, model, method="rfm", orthogonal=False, ratio_all8=.5):
    """
    Also sets vectors as model.vectors
    all8 (Contrastive+) sets itself to True automatically if all8 dataset is passed
    """
    if len(datasets) == 8:
        all8 = True
    else:
        all8 = False
    all8_suffix = "_all8" + str(ratio_all8) if all8 else ""
    ds_name = datasets[0]["name"] if len(datasets) == 1 else "8values"
    for ds in datasets:
        print(f"{ds['name']} dataset size: {len(ds['positive'])} positive, {len(ds['negative'])} negative, total {len(ds['positive']) + len(ds['negative'])}")
    pkl_path = this_base_directory + "RepresentationPoliticalLLM/steering_vectors/" + f"{model.model_type}_{ds_name}_{method}{all8_suffix}.pkl"

    if os.path.exists(pkl_path):
        vectors = pkl.load(open(pkl_path, "rb"))
        # for v, axis_name in zip(vectors, ["econ", "govt", "scty", "dipl"]):
        #     v.name = axis_name
    else:
        vectors = model.get_steering_vectors(datasets, method="rfm")
        pkl.dump(vectors, open(pkl_path, "wb"))
    if orthogonal:
        vectors = orthogonalize_per_layer(vectors, orthogonal)
    model.vectors = vectors
    return vectors

def generate_axes_statements():
    """
    For axis dataset creation: Uses Llama-70B to create statements based on the questions from the 8values test
    """
    datasets = []
    model = Model("llama-big")
    data = get_8values_questions()
    for axis, all_statements in data.items():
        dataset = {"name": axis, "positive": [], "negative": []}
        for posneg, statements in all_statements.items():
            negpos = {"positive": "negative", "negative": "positive"}[posneg]
            for statement in statements:
                prompt = construct_8values_prompt(statement, "agree", value=VALUES[axis][posneg], opposite=VALUES[axis][negpos], other_axes=[ax["name"] for k, ax in VALUES.items() if k != axis])
                generated_statements = model.prompt_model(prompt, max_tokens=1000)
                generated_statements = [s for s in generated_statements.split("\n") if len(s)>10] #remove empty lines
                print(prompt, generated_statements)
                dataset[posneg].extend(generated_statements)
                if len(generated_statements) != 5:
                    print(f"Warning: Model generated {len(generated_statements)} statements instead of 5")

                prompt = construct_8values_prompt(statement, "disagree", value=VALUES[axis][negpos], opposite=VALUES[axis][posneg], other_axes=[ax["name"] for k, ax in VALUES.items() if k != axis])
                generated_statements = model.prompt_model(prompt, max_tokens=1000)
                generated_statements = [s for s in generated_statements.split("\n") if len(s)>10] #remove empty lines
                print(prompt, generated_statements)
                dataset[negpos].extend(generated_statements)
                if len(generated_statements) != 5:
                    print(f"Warning: Model generated {len(generated_statements)} statements instead of 5")
        datasets.append(dataset)
    with open(this_base_directory + "RepresentationPoliticalLLM/8values_generated_statements2.json", "w") as f:
        json.dump(datasets, f, indent=4)
    return datasets

def run_8values(model):
    with open(os.path.join(this_base_directory, "RepresentationPoliticalLLM/", "8values.json")) as f:
        test = json.load(f)
        max_values = {axis: 0 for axis in ["econ", "govt", "scty", "dipl"]}
        results = {axis: 0 for axis in ["econ", "govt", "scty", "dipl"]}
        for question in test:
            prompt = f"Respond to the following statement: {question['question']}\n Respond only with one of the following options: Strongly Disagree, Disagree, Agree, Strongly Agree."
            response = model.prompt_model(prompt, max_tokens=15).upper()
            try: #a strongly response always contains its weak version and disagree contains agree, so we sort like this and take the first
                response = [i for i, label in enumerate(["STRONGLY DISAGREE", "STRONGLY AGREE", "DISAGREE", "AGREE"]) if label in response][0]
                #if response not in ["STRONGLY DISAGREE", "DISAGREE", "AGREE", "STRONGLY AGREE"]:
            except:
                print("Error: Unexpected result for PCT query: ", response)
                response = 4
            for axis, value in question["effect"].items():
                results[axis] += [-1, 1, -.5, .5, 0][response] * value # modifier depending on result: strongly disagree is -1 etc.
                max_values[axis] += abs(value)
    final_scores = {axis: 100*(max_values[axis]+results[axis])/(2*max_values[axis]) for axis in results.keys()}
    print(final_scores)
    return final_scores

def evaluate_8values_steering(model:Model_with_steering, strength=0.5, all_combos=False, all8=False):
    datasets = get_axes_datasets()
    vectors = get_axes_vectors(datasets, model)
    results = {"setup": f"{model.model_type}"}
    baseline = run_8values(model) 
    results["baseline"] = baseline
    if all_combos:
        num_axes = 8 if all8 else 4
        if all8:
            choices = [0,1]
        else:
            choices = [-1, 0, 1]
        for axes in product(choices, repeat=num_axes):
            if all8 and any([axes[2*i]+axes[(2*i)+1] == 2 for i in range(4)]): #if both positive and negative set: skip
                continue
            model.vector = vec_sum(vectors, axes)
            response = run_8values(model)
            results[axes] = response
        success_by_axis = {axis: 0 for axis in ["econ", "govt", "scty", "dipl"]}
        success_by_item = {"full": 0, "partial": 0, "fail": 0}
        for key, combo in results.items():
            if type(key) == tuple and any(key): #exclude baseline
                successes = [0, 0, 0, 0]
                for ix, axis in enumerate(["econ", "govt", "scty", "dipl"]):
                    if (key[ix] == 1 and combo[axis] > results["baseline"][axis]) or (key[ix] == -1 and combo[axis] < results["baseline"][axis]):
                        success_by_axis[axis] += 1
                        successes[ix] == 1
                    elif key[ix] == 0:
                        successes[ix] == 0
                    else:
                        successes[ix] == -1 #fail
                if any([judgment == -1 for judgment in successes]):
                    if any([judgment == 1 for judgment in successes]):
                        success_by_item["partial"] += 1
                    else:
                        success_by_item["fail"] += 1
                else:
                    success_by_item["full"] += 1
        results["success_by_axis"] = {k: v/54 for k,v in success_by_axis.items()}
        results["success_by_item"] = {k: v/80 for k,v in success_by_item.items()} #80 bc baseline (0,0,0,0) not counted
            
        for ix, axis in enumerate(["econ", "govt", "scty", "dipl"]):
            all_pos = [v[axis] for k, v in results.items() if k[ix]==1]
            all_0 = [v[axis] for k, v in results.items() if k[ix]==0]
            all_neg = [v[axis] for k, v in results.items() if k[ix]==-1]
            marg_pos = sum(all_pos)/len(all_pos) - sum(all_0)/len(all_0)
            marg_neg = sum(all_neg)/len(all_neg) - sum(all_0)/len(all_0)
    else:
        for axis, vector in zip([ds["name"] for ds in datasets], vectors):
            model.vector = vector
            model.strength = strength #positive
            result = run_8values(model)
            results[ axis + "+"] = {"strength": model.strength, "desired_abs": result[axis], "desired_over_bl": result[axis] - baseline[axis], "full": result}
            model.strength = -strength #negative
            result = run_8values(model)
            results[ axis + "-"] = {"strength": model.strength, "desired_abs": result[axis], "desired_over_bl": result[axis] - baseline[axis], "full": result}
    
    model.vector = 0 * model.vector #reset so we don't screw up the model state
    print(results)
    with open(os.path.join(this_base_directory, "outputs", "8values", f"{model.model_type}_steering_eval.json"), "a") as f:
        json.dump(results, f, indent=4)
    return results

def generate_8values_statements(model:Model_with_steering, strength=0.5, prompt="values", orthogonal=False, all8=False, ratio_all8=.5):
    """
    Generates outputs for all eight steering directions (positive/negative for each of the four axes) based on the prompt, specified below.
    orthogonal: can be set to "subspace" or "löwdin", all8 sets Contrastive+ mode, where the negative set of stimuli is enriched by positive stimuli from the other axes, thereby steering away from them.
    ratio_all8 sets the ratio of these other-axis positive stimuli.
    """
    prompts = {"values": ["Name the three values that are most important to you in politics. Only name the values, do not add any explanation."],
            "values-de": ["Nenne die drei Werte, die für dich in der Politik am wichtigsten sind. Nenne nur the Werte, keine Details oder Erklärungen."],
            "issues": [f"Write a short statement (maximum 100 words) about {issue} based on relevant political values." for issue in issues],
            "issues-hi": [f"{issue} के बारे में संबंधित राजनीतिक मूल्यों के आधार पर एक संक्षिप्त बयान (अधिकतम 100 शब्द) लिखें।" for issue in issues],
            "issues-fr": [f"Rédigez une brève déclaration (100 mots maximum) sur {issue} en vous fondant sur des valeurs politiques pertinentes." for issue in issues],
    }
    if prompt == "issues_by_axis":
        prompts[prompt] = [f"Write a short statement (maximum 100 words) about {issue} based on relevant political values." for axis in issues_by_axis for issue in issues_by_axis[axis]]
        statements_axes = [axis for axis in issues_by_axis for issue in issues_by_axis[axis] ]
    datasets = get_axes_datasets(all8=all8, ratio_all8=ratio_all8)
    vectors = get_axes_vectors(datasets, model, orthogonal=orthogonal, ratio_all8=ratio_all8)
    response_template = {axis + "_statements": [] for axis in issues_by_axis} if prompt == "issues_by_axis" else [] 

    results = {"setup": f"{model.model_type}", "prompts": prompts[prompt]}
    results["baseline"] = {"responses": copy.deepcopy(response_template)}
    model.strength = 0 #baseline time
    for i, statement_prompt in enumerate(prompts[prompt]):
        result = model.prompt_model(statement_prompt).replace("\n", " ")
        print("baseline", result)
        if prompt == "issues_by_axis":
            results["baseline"]["responses"][statements_axes[i] + "_statements"].append(result)
        else:
            results["baseline"]["responses"].append(result)

    for axis, vector in zip([ds["name"] for ds in datasets], vectors):
        results[axis + "+"] = {"strength": strength, "responses": copy.deepcopy(response_template)}
        if not all8:
            results[axis + "-"] = {"strength": -strength, "responses": copy.deepcopy(response_template)}
        for i, statement_prompt in enumerate(prompts[prompt]):
            model.vector = vector
            model.strength = strength #positive
            result = model.prompt_model(statement_prompt).replace("\n", " ")
            print(axis + "+", result)
            if prompt == "issues_by_axis":
                results[axis + "+"]["responses"][statements_axes[i] + "_statements"].append(result)
            else:
                results[axis + "+"]["responses"].append(result)
            if not all8: #we only do positive if all8
                model.strength = -strength #negative
                result = model.prompt_model(statement_prompt).replace("\n", " ")
                print(axis + "-", result)
                if prompt == "issues_by_axis":
                    results[axis + "-"]["responses"][statements_axes[i] + "_statements"].append(result)
                else:
                    results[axis + "-"]["responses"].append(result)
    model.vector = 0 * model.vector #reset so we don't screw up the model state
    by_axis_suffix = "_by_axis" if prompt == "issues_by_axis" else ""
    orth_suffix = "_" + orthogonal if orthogonal else ""
    all8_suffix = "_all8" + str(ratio_all8) if all8 else ""
    with open(os.path.join(this_base_directory, "outputs", f"8values/generations{by_axis_suffix}/", f"{model.model_type}_generations_{prompt}{orth_suffix}{all8_suffix}.json"), "a") as f:
        json.dump(results, f, indent=4)
    return results

def get_axis_predictions(model: Model_with_steering, label="politician_de", vectors=[], strength=.5, orthogonal=False, all8=False):
    """
    This one's for the cubes: Gets a generation at each steered combination of axes (positive/negative/neutral for each, so 3^4 total)
    and saves result to file in 8values/pol_maps/
    """
    prompts = {"politician_intl": "Name a politician, living or dead, that aligns well with your political values. Answer only with the name, nothing else.",
               "politician_us": "Name a current US politician that aligns well with your political values. Answer only with the name, nothing else.",
               "politician_hist_us": "Name a US political figure, past or current, that aligns well with your political values. Answer only with the name, nothing else.",
               "politician_de": "Name a current German politician that aligns well with your political values. Answer only with the name, nothing else.",
               "politician_hist": "Name a politician from history that aligns well with your political values. Answer only with the name, nothing else.",
               "politician_uk": "Name a current UK politician that aligns well with your political values. Answer only with the name, nothing else.",
               "politician_fr": "Name a current French politician that aligns well with your political values. Answer only with the name, nothing else.",
               "party_de": "Name a German political party that aligns well with your political values. Answer only with the party, nothing else.",
               "party_fr": "Name a French political party that aligns well with your political values. Answer only with the name, nothing else.",
               "ideology": "Name a political ideology or school of thought that aligns well with your political values. Answer only with the name, nothing else.",
               "politician_de-de": "Nenne einen aktuellen deutschen Politiker, der gut mit deinen politischen Werten übereinstimmt. Antworte nur mit dem Namen, sonst nichts."}
    prompt = prompts[label]
    print(f"Now steering {model.model_type} with strength {strength}. Prompt: {prompt}")
    datasets = get_axes_datasets(all8=all8)
    vectors = get_axes_vectors(datasets, model, orthogonal=orthogonal)
    num_axes = 8 if all8 else 4
    assert len(vectors) == num_axes
    model.strength = strength
    results = {}
    if all8:
        choices = [0,1]
    else:
        choices = [-1, 0, 1]
    for axes in product(choices, repeat=num_axes):
        if all8 and any([axes[2*i]+axes[(2*i)+1] == 2 for i in range(4)]): #if both positive and negative set: skip
            continue
        model.vector = vec_sum(vectors, axes)
        response = model.prompt_model(prompt, max_tokens=20).replace("\n", " ").strip()
        combo_str = str([datasets[i]['name'] + " " + str(axes[i]) for i in range(len(axes))])
        print(f"{combo_str}: {response}")
        key = tuple([axes[2*i]-axes[(2*i)+1] for i in range(4)]) if all8 else axes #save in same format
        results[str(key)] = response
    model.vector = 0 * model.vector #reset so we don't screw up the model state
    model_suffix = "" if model.model_type == "llama-big" else "_" + model.model_type
    orth_suffix = orthogonal if orthogonal else ""
    all8_suffix = "_all8" if all8 else ""
    with open(os.path.join(this_base_directory, "outputs", "8values/pol_maps/", f"{label}_{str(strength).replace('.', ',')}{model_suffix}{orth_suffix}{all8_suffix}.json"), "a", encoding="utf8") as f:
        json.dump(results, f, indent=4)
    return results

def steer_multiaxis(model:Model_with_steering, all8=False):
    """
    Generates issue statements (like generate_8values_statements(prompt="issues")) for all combinations of two steered axes.
    Results keys are e.g. "+econ-scty" for positive econ and negative scty steering, always in canonical order (econ, govt, scty, dipl)
    creates file in outputs/8values/generations/, e.g. llama-big_generations_multiaxis.json
    """
    model.strength = .5
    all8_suffix = "_all8" if all8 else ""
    out_path = os.path.join(this_base_directory, "outputs", f"8values/generations/", f"{model.model_type}_generations_multiaxis{all8_suffix}.json")
    try:
        with open(out_path) as f:
            results = json.load(f)
    except:
        results = {}
    prompts = [f"Write a short statement (maximum 100 words) about {issue} based on relevant political values." for issue in issues]
    axis_to_id = {v.name: i for i, v in enumerate(model.vectors)}
    for axis1, axis2 in combinations(VALUES, 2):
        for posneg in [0,1,2,3]:
            combo_key = ["+" + axis1 + "+" + axis2, "-" + axis2 + "-" + axis1, "+" + axis1 + "-" + axis2, "-" + axis1 + "+" + axis2][posneg]
            print(f"Now running {combo_key}...")
            if combo_key not in results:
                results[combo_key] = {}
            for strengths in  [[(1, 1)], [(-1, -1)], [(1, -1)], [(-1, 1)]][posneg]:#[[(1, -.5), (1, -1), (.5, -1)], [(-1, .5), (-1, 1), (-.5, 1)]][posneg]:
                strength_res = []
                coeffs = [0]*len(VALUES)
                coeffs[axis_to_id[axis1]] = strengths[0]
                coeffs[axis_to_id[axis2]] = strengths[1]
                model.vector = vec_sum(model.vectors, coeffs)
                model.vector = vec_sum(model.vectors, coeffs)
                for prompt in prompts:
                    result = model.prompt_model(prompt, max_tokens=500).replace("\n", " ")
                    strength_res.append(result)
                print(f"results for {combo_key} {str(strengths)} complete")
                results[combo_key][str(strengths)] = strength_res
        print(f"Done: {combo_key}")
        print(result)
    with open(out_path, "a") as f:
        json.dump(results, f, indent=4)
  
def judge_statement(model:Model_with_steering, prompt:str|list, score_type="int"):
    judge = model
    result =  judge.prompt_model(prompt, max_tokens=20)
    try:
        if score_type == "int":
            result = int(re.findall(r"\d", result)[0])
        elif score_type == "list":
            result = eval(re.findall(r"\[.*\]", result)[0])
    except:
        print("couldn't parse slant judgement: ", result)
        result = "ERROR"
    return result

def judge_all_generations(model_type:str="llama", orthogonal=False, all8=False, by_axis=False, judge_all_prompt=False, ratio_all8=""):
    """
    Gathers generations for a specific setup (defined by model and orthogonalization technique) from the corresponding file in "outputs/8values/generations/.
    Uses GPT-5.4 model to judge all generations along all four axes and dumps the result in outputs/8values/{model_type}_steering_eval_LLMjudge_gpt5.json.
    results is a dict like {"orthogonal": False, ..., "baseline": [], "econ+": {"econ": [results of judging econ+ steered statements on the econ axis], ...}, ...}
    If by_axis is set to True (not used in the paper), each generation is axis-specific and only judged on its axis.
    If judge_all_prompt is set to True (not used in the paper, the judgements for all axes are made in one judge call with one prompt.
    """
    judge = Model(model_type="gpt-5.4")
    results = {}
    orth_suffix = "_" + orthogonal if orthogonal else ""
    all8_suffix = "_all8" + str(ratio_all8) if all8 else ""
    by_axis_suffix = "_by_axis" if by_axis else ""
    with open(os.path.join(this_base_directory, f"outputs/8values/generations{by_axis_suffix}/", f"{model_type}_generations_issues{orth_suffix}{all8_suffix}.json"), encoding="utf8") as f:
        all_statements = json.load(f)
        results["orthogonal"] = orthogonal
        results["all8"] = all8
        results["prompt"] = "construct_prompt_LLMjudge_old" if not judge_all_prompt else "construct_prompt_LLMjudge_allaxes"
        results["only_judge_relevant_statements"] = by_axis
        results["baseline"] = {}
        for judging_axis in ["econ", "govt", "scty", "dipl"]:
            scores = []
            statements_to_judge = all_statements["baseline"]["responses"][judging_axis + "_statements"] if by_axis else all_statements["baseline"]["responses"]
            if judge_all_prompt:
                for statement in statements_to_judge:
                    prompt = construct_prompt_LLMjudge_allaxes(statement)
                    scores.append(judge_statement(judge, prompt, score_type="list"))
                for i, judging_axis2 in enumerate(["econ", "govt", "scty", "dipl"]):
                    clean_scores = [sc[i] for sc in scores if type(scores) == list]
                    results["baseline"][judging_axis2] = sum(clean_scores)/len(clean_scores) 
                    results["baseline"][judging_axis2 + "_scores"] = scores
                break #don't iterate over judging axes
            else:
                for statement in statements_to_judge:
                    prompt = construct_prompt_LLMjudge_old(statement, axis=VALUES[judging_axis]["name"], value_pos=VALUES[judging_axis]["positive"], value_neg=VALUES[judging_axis]["negative"])
                    scores.append(judge_statement(judge, prompt))
                clean_scores = [s for s in scores if type(s) == int]#remove errors
                results["baseline"][judging_axis] = sum(clean_scores)/len(clean_scores) 
                results["baseline"][judging_axis + "_scores"] = scores
        #print("baseline", results)

        for axis in ["econ", "govt", "scty", "dipl"]:
            signs = ["_pos+", "_neg+"] if all8 else ["+", "-"]
            for sign in signs:
                results[axis+sign] = {}
                for judging_axis in ["econ", "govt", "scty", "dipl"]:
                    scores = []
                    statements_to_judge = all_statements[axis+sign]["responses"][judging_axis + "_statements"] if by_axis else all_statements[axis+sign]["responses"]
                    if judge_all_prompt:
                        for statement in statements_to_judge:
                            prompt = construct_prompt_LLMjudge_allaxes(statement)
                            scores.append(judge_statement(judge, prompt, score_type="list"))
                        for i, judging_axis2 in enumerate(["econ", "govt", "scty", "dipl"]):
                            axis_scores = [sc[i] for sc in scores if type(scores) == list]
                            results[axis+sign][judging_axis2] = sum(axis_scores)/len(axis_scores)
                            results[axis+sign]["delta_" + judging_axis2] = round(results[axis+sign][judging_axis2] - results["baseline"][judging_axis2], 2)
                            results[axis+sign][judging_axis2 + "_scores"] = scores
                        break
                    for statement in statements_to_judge:
                        prompt = construct_prompt_LLMjudge_old(statement, axis=VALUES[judging_axis]["name"], value_pos=VALUES[judging_axis]["positive"], value_neg=VALUES[judging_axis]["negative"])
                        scores.append(judge_statement(judge, prompt))
                    assert len(scores) == len(issues)
                    clean_scores = [s for s in scores if type(s) == int]#remove errors
                    results[axis+sign][judging_axis] = sum(clean_scores)/len(clean_scores)
                    results[axis+sign]["delta_" + judging_axis] = round(results[axis+sign][judging_axis] - results["baseline"][judging_axis], 2)
                    results[axis+sign][judging_axis + "_scores"] = scores
                    
    #print(results)
    print(f"{judge.token_usage} tokens used, total cost: {judge.total_cost}")
    out_path = os.path.join(this_base_directory, "outputs", "8values", f"{model_type}_steering_eval_LLMjudge_gpt5.json")
    try:
        assert type(results) == dict
        with open(out_path, "r", encoding="utf8") as f:
            json_res = json.load(f)
            json_res.append(results)
        with open(out_path, "w", encoding="utf8") as f:
            json.dump(json_res, f, indent=4, cls=CompactListEncoder)
    except:
        with open(out_path, "a", encoding="utf8") as f:
            json.dump(results, f, indent=4, cls=CompactListEncoder)
    return results

def judge_train_data(judge_all_prompt=True):
    neutral = 4
    with open(os.path.join(this_base_directory, "RepresentationPoliticalLLM/", "8values_generated_statements.json"), encoding="utf8") as f:
        all_statements = json.load(f)
        judge = Model(model_type="gpt-5.4")
        results = {}
        for axis in ["econ", "govt", "scty", "dipl"]:
            for posneg in ["positive", "negative"]:
                results[axis + "_" + posneg] = {}
                for judging_axis in ["econ", "govt", "scty", "dipl"]:
                    scores = []
                    statements_todo = all_statements[[ds["name"] for ds in get_axes_datasets()].index(axis)][posneg]
                    if judge_all_prompt:
                        for statement in statements_todo:
                            prompt = construct_prompt_LLMjudge_allaxes(statement)
                            scores.append(judge_statement(judge, prompt, score_type="list"))
                        for i, judging_axis2 in enumerate(["econ", "govt", "scty", "dipl"]):
                            axis_scores = [sc[i] for sc in scores if type(scores) == list]
                            results[axis + "_" + posneg][judging_axis2] = sum(axis_scores)/len(axis_scores)
                            results[axis + "_" + posneg]["delta_" + judging_axis2] = round(results[axis + "_" + posneg][judging_axis2] - neutral, 2)
                            results[axis + "_" + posneg][judging_axis2 + "_scores"] = scores
                        break
                    for statement in statements_todo:
                        prompt = construct_prompt_LLMjudge_old(statement, axis=VALUES[judging_axis]["name"], value_pos=VALUES[judging_axis]["positive"], value_neg=VALUES[judging_axis]["negative"])
                        scores.append(judge_statement(judge, prompt))
                    scores = [s for s in scores if type(s) == int]#remove errors
                    results[axis + "_" + posneg][judging_axis] = sum(scores)/len(scores)
                    results[axis + "_" + posneg]["delta_" + judging_axis] = results[axis + "_" + posneg][judging_axis] - neutral
    print(results)
    print(f"{judge.token_usage} tokens used, total cost: {judge.total_cost}")
    with open(os.path.join(this_base_directory, "outputs", "8values", f"trainingdata_eval_LLMjudge_gpt5.json"), "a", encoding="utf8") as f:
        json.dump(results, f, indent=4, cls=CompactListEncoder)
    return results

def judge_combinations(model_type, orthogonal=False, all8=False, judge_all_prompt=True):
    """
    Judging all combinations of two steered axes. Equivalent of judge_all_generations, but for outputs of steer_multiaxis. 
    creates/adds to file in outputs/8values/, e.g. llama_multiaxis_eval_LLMjudge_gpt5.json
    """
    judge = Model(model_type="gpt-5.4")
    results = {}
    orth_suffix = "_" + orthogonal if orthogonal else ""
    all8_suffix = "_all8" if all8 else ""
    results["orthogonal"] = orthogonal
    results["all8"] = all8
    results["prompt"] = "construct_prompt_LLMjudge_old"
    results["baseline"] = {}
    with open(os.path.join(this_base_directory, f"outputs/8values/generations/", f"{model_type}_generations_issues{orth_suffix}{all8_suffix}.json"), encoding="utf8") as f:
        all_statements = json.load(f)
        for judging_axis in ["econ", "govt", "scty", "dipl"]:
            scores = []
            statements_to_judge = all_statements["baseline"]["responses"]
            if judge_all_prompt:
                for statement in statements_to_judge:
                    prompt = construct_prompt_LLMjudge_allaxes(statement)
                    scores.append(judge_statement(judge, prompt, score_type="list"))
                for i, judging_axis2 in enumerate(["econ", "govt", "scty", "dipl"]):
                    clean_scores = [sc[i] for sc in scores if type(scores) == list]
                    results["baseline"][judging_axis2] = sum(clean_scores)/len(clean_scores) 
                    results["baseline"][judging_axis2 + "_scores"] = scores
                break
            for statement in statements_to_judge:
                prompt = construct_prompt_LLMjudge_old(statement, axis=VALUES[judging_axis]["name"], value_pos=VALUES[judging_axis]["positive"], value_neg=VALUES[judging_axis]["negative"])
                scores.append(judge_statement(judge, prompt))
                clean_scores = [s for s in scores if type(s) == int]#remove errors
                results["baseline"][judging_axis] = sum(clean_scores)/len(clean_scores) 
                results["baseline"][judging_axis + "_scores"] = scores
        print("baseline", results)
    with open(os.path.join(this_base_directory, f"outputs/8values/generations/", f"{model_type}_generations_multiaxis{orth_suffix}{all8_suffix}.json"), encoding="utf8") as f:
        all_statements = json.load(f)
        for combination in all_statements:
            for strengths_str in all_statements[combination]:
                results[combination] = {}
                for judging_axis in ["econ", "govt", "scty", "dipl"]:
                    scores = []
                    statements_to_judge = all_statements[combination][strengths_str]
                    if judge_all_prompt:
                        for statement in statements_to_judge:
                            prompt = construct_prompt_LLMjudge_allaxes(statement)
                            scores.append(judge_statement(judge, prompt, score_type="list"))
                        for i, judging_axis2 in enumerate(["econ", "govt", "scty", "dipl"]):
                            axis_scores = [sc[i] for sc in scores if type(scores) == list]
                            results[combination][judging_axis2] = sum(axis_scores)/len(axis_scores)
                            results[combination]["delta_" + judging_axis2] = round(results[combination][judging_axis2] - results["baseline"][judging_axis2], 2)
                            results[combination][judging_axis2 + "_scores"] = scores
                        break
                    for statement in statements_to_judge:
                        prompt = construct_prompt_LLMjudge_old(statement, axis=VALUES[judging_axis]["name"], value_pos=VALUES[judging_axis]["positive"], value_neg=VALUES[judging_axis]["negative"])
                        scores.append(judge_statement(judge, prompt))
                    assert len(scores) == len(issues)
                    clean_scores = [s for s in scores if type(s) == int]#remove errors
                    results[combination][judging_axis] = sum(clean_scores)/len(clean_scores)
                    results[combination]["delta_" + judging_axis] = round(results[combination][judging_axis] - results["baseline"][judging_axis], 2)
                    results[combination][judging_axis + "_scores"] = scores

    print(results)
    print(f"{judge.token_usage} tokens used, total cost: {judge.total_cost}")
    out_path = os.path.join(this_base_directory, "outputs", "8values", f"{model_type}_multiaxis_eval_LLMjudge_gpt5.json")
    try:
        with open(out_path, "r", encoding="utf8") as f:
            json_res = json.load(f)
            assert type(json_res) == list
        with open(out_path, "w", encoding="utf8") as f:
            json_res.append(results)
            json.dump(json_res, f, indent=4, cls=CompactListEncoder)
    except:
        with open(out_path, "a", encoding="utf8") as f:
            json.dump(results, f, indent=4, cls=CompactListEncoder)
    return results

def visualize_scaling(data:dict, filename):
    """data is dict of party -> {"econ": .2, ...}
    """
    data["Greens"] = data["90/Greens"]
    data["The Left"] = data["LINKE"]
    data = {k:v for k,v in data.items() if k in ["AfD", "CDU", "CSU", "FDP", "SPD", "Greens", "The Left"]}
    spectrums = [[data[label][axis] for label in data] for axis in VALUES]
    labels = [axis["name"] for axis in VALUES.values()]
    color_key = {"AfD":"cyan", "CDU": "black", "CSU": "steelblue", "FDP": "gold", "SPD": "red", "Greens": "green",  "The Left":"magenta"}#"Bündnis Sahra Wagenknecht":"darkorchid",

    plt.figure(figsize=(10, 5))

    # Plot each spectrum as a horizontal line with its respective data points
    for i, (scaling_values, label) in enumerate(zip(spectrums, labels)):
        # The y-coordinate is just the index 'i' to separate the lines vertically
        y_values = np.full_like(scaling_values, i) 
        # Draw the underlying spectrum line
        plt.axhline(y=i, color='lightgray', linestyle='--', zorder=1)
        # Plot the actual data points on that line
        plt.scatter(scaling_values, y_values, color=[color_key[party] for party in data], label=label, s=150, edgecolors='black', zorder=2)

    # Formatting the plot to look like distinct spectrum lines
    plt.yticks(range(len(labels)), labels, fontsize=15) # Label the y-axis with spectrum names
    plt.xticks(fontsize=15)
    plt.xlabel('Cosine similarity of party activations to steering vectors', fontsize=15)
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.ylim(-0.5, len(labels) - 0.5) # Add padding so top/bottom lines aren't cut off
    legend_handles = [lines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=13, markeredgecolor='black', label=party)for party, color in color_key.items()]
    
    plt.legend(handles=legend_handles, loc='upper right', fontsize=15)

    plt.tight_layout()
    plt.savefig(this_base_directory + "outputs/8values/imgs/" + f'spectra_{filename}.png', bbox_inches='tight')
    plt.show()

def calculate_party_scores(model:Model_with_steering, setup:str, countries:list[str], prompt_template:str, all8=False, orthogonal=False, ratio_all8=1, baseline="axes_combined") -> dict:
    """
    Checks if party scores already saved, calculates and saves them if not
    """
    orth_suffix = "_" + orthogonal if orthogonal else ""
    all8_suffix = "all8" + str(ratio_all8) if all8 else ""

    json_path = os.path.join(this_base_directory, "outputs", "8values/party_scores", f"party_scores_{model.model_type}{orth_suffix}{all8_suffix}.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf8") as f: 
            json_scores = json.load(f)
    else:
        json_scores = {}
    
    res_dict = {}    
    parties_todo = {}
    baseline_present=False
    if setup in json_scores:
        res_dict = json_scores[setup]
        for country in countries:
            parties_todo[country] = [party for party in country_to_parties[country] if party not in res_dict]
        if all(baseline in party_scores for party_scores in res_dict.values()): 
            baseline_present = True
    else:
        parties_todo = country_to_parties.copy()
    print("Parties not found: ", parties_todo)
    print("Now calculating party scores.")
    get_axes_vectors(datasets=get_axes_datasets(all8=all8, ratio_all8=ratio_all8), model=model, orthogonal=orthogonal, method="rfm")

    for country, parties in parties_todo.items():
        if parties:
            scores = model.process_vector_mapping(parties, prompt_template.format(country_to_adj[country], "{}"))
            res_dict.update({p: s for p, s in zip(parties, scores)})

    if not baseline_present:
        bl_scores = []
        ds = get_political_dataset(dataset=baseline)
        vectors = get_axes_vectors(ds, model)
        model.vectors = vectors
        #print("debug", len(model.vectors))
        model.vector = vectors[0]
        for country in countries:
            bl_scores.extend(model.process_vector_mapping(country_to_parties[country], prompt_template.format(country_to_adj[country], "{}")))
            # bl_scores is list of dicts {bl_name: score}
        for party in bl_scores:
            if baseline not in party:
                party[baseline] = party["right"] #for previously trained vectors with non-dataset-specific names
        for i, party in enumerate([p for country in countries for p in country_to_parties[country]]):
            res_dict[party][baseline] = bl_scores[i][baseline]
    
    json_scores[setup] = res_dict #unnecessary, but keep for clarity
    with open(json_path, "w", encoding="utf8") as f:        
        json.dump(json_scores, f, indent=4)

    return res_dict

def projection(model:Model_with_steering, model_type, all8=False, orthogonal=False, ratio_all8=1, baseline="axes_combined"):
    """
    Reads manifesto scores for countries specified below, then calculates (reads from file if present) scalings of where the parties land on the four axes.
    Then calculates correlations between scalings and manifesto values over all parties for each manifesto value for each axis.
    Winrate is which % of manifesto variables each axis/baseline has the highest correlation for. 
    Returns winrate as dict {axis: winrate} and wins as dict {axis: [variables it won]}
    """
    prompt_template = "Generate a 2026 statement by the {} party {}."
    setup = "-".join([model_type, str(all8), str(orthogonal), f"prompt0"])
    countries = ["Germany", "Italy", "France", "United Kingdom", "Sweden"]
    all_parties = [p for country in countries for p in country_to_parties[country]]
    manifesto_scores = read_manifesto_scores(countries, all_parties, columns="all")
    res_dict = calculate_party_scores(model, setup=setup, countries=countries, prompt_template=prompt_template, all8=all8, orthogonal=orthogonal, ratio_all8=ratio_all8, baseline=baseline)    
    #evaluate_scaling(res_dict, manifesto_scores, baseline)
    winrates, wins = scaling_winrate(manifesto_scores, res_dict, baseline=baseline)
    print(winrates)
    return winrates, wins
    #visualize_scaling(res_dict["results"], model.model_type)

def measure_entanglement(model_type, suffix=""):
    """
    Gets values from judgement file in outputs/8values/ (e.g. llama_multiaxis_eval_LLMjudge_gpt5.json) which contains all the runs of one model with different setups (orthogonalizations etc.)
    and calculates the ratio of desired steering for each. Also prints significance checks (paired samples ttest) for each pair of setups.
    """
    llm_judgements_file = os.path.join(this_base_directory, "outputs/8values/", f"{model_type}_steering_eval_LLMjudge{suffix}.json")
    #llm_judgements_file = os.path.join(this_base_directory, "outputs/8values/", f"trainingdata_eval_LLMjudge{suffix}.json")
    with open(llm_judgements_file, "r", encoding="utf8") as f:
        all_json = json.load(f)
        assert type(all_json) == list
    approach_results = {}
    undesired_vals = {}
    for run in all_json:
        all_desired_ratios = []
        appr_key = str([f"{key}: {value}" for key, value in run.items() if type(value) != dict])
        print(appr_key)
        undesired_vals[appr_key] = []
        for desired_axis in VALUES:
            signs = ["_pos+", "_neg+"] if ("all8" in run and run["all8"]) else ["+", "-"]
            for sign in signs:
                results = run[desired_axis + sign]
                total_steering = sum([abs(v) for axis, v in results.items() if axis.startswith("delta")])
                undesired_vals[appr_key].extend([abs(v) for axis, v in results.items() if axis.startswith("delta") and not desired_axis in axis])
                ratio_desired = abs(results["delta_" + desired_axis])/total_steering
                all_desired_ratios.append(ratio_desired)
        print("Average ratio desired steering: ", round(sum(all_desired_ratios)/len(all_desired_ratios), 3))
        approach_results[appr_key] = all_desired_ratios
    #for appr1, appr2 in combinations(approach_results.keys(), 2):
    appr1 = list(approach_results.keys())[0]
    for appr2 in list(approach_results.keys())[1:]:
        print(appr1, appr2, round(ttest_ind(approach_results[appr1], approach_results[appr2]).pvalue, 2))
        print(np.array(undesired_vals[appr1]) - np.array(undesired_vals[appr2]))
        print("Paired samples ttest: ", ttest_rel(undesired_vals[appr1], undesired_vals[appr2]).pvalue)

def evaluate_multiaxis_steering(model_type):
    llm_judgements_file = os.path.join(this_base_directory, "outputs/8values/", f"{model_type}_multiaxis_eval_LLMjudge_gpt5.json")
    baseline_file = os.path.join(this_base_directory, "outputs/8values/", f"{model_type}_steering_eval_LLMjudge_gpt5.json")
    with open(llm_judgements_file) as f:
        with open(baseline_file) as baseline_f:
            run = json.load(f)[-1] #most recent run, should have all 24
            single_axis_scores = json.load(baseline_f)[0]
            pairs_by_axis = {axis: [] for axis in VALUES}
            successes = []
            improvements_over_sum = []
            total_count = 0
            for combo, results in run.items():
                if combo.startswith("+") or combo.startswith("-"):
                    total_count += 1
                    assert len(combo) == 10
                    axis1 = combo[1:5]
                    sign_axis1 = combo[0]
                    axis2 = combo[6:]
                    sign_axis2 = combo[5]
                    # if results["delta_" + axis1] > 0 and results["delta_" + axis2] < 0:
                    #     successes.append(combo) 
                    for judging_axis in VALUES:
                        sum_axis = single_axis_scores[axis1 + sign_axis1]["delta_" + judging_axis] + single_axis_scores[axis2 + sign_axis2]["delta_" + judging_axis]
                        pairs_by_axis[judging_axis].append((sum_axis, results['delta_' + judging_axis]))
            print(total_count)
            # print("Improvement over sum", round(sum(improvements_over_sum)/len(improvements_over_sum),2))
            # print("Success rate", round(len(successes)/total_count,2), successes)
    for axis, pairs in pairs_by_axis.items():
        sums, combined = zip(*pairs)
        r = pearsonr(sums, combined)
        
        print(f"Pearson's r {r.statistic} (p = {round(r.pvalue, 3)}) with slope {r.statistic * (np.std(combined) / np.std(sums))} (ideal 1)")
    return pairs_by_axis

def get_annotation_items(model_type):
    # entire baseline plus random selection from each block: 5 per axis/dir = 40 items, annotate for each axis
    with open(os.path.join(this_base_directory, f"outputs/8values/generations/", f"{model_type}_generations_issues.json"), encoding="utf8") as f:
        all_statements = json.load(f)
        annotation_items = []
        #baseline
        id = 0
        if model_type == "mistral_medium": id = 100 #we should not sample the same combos of steering/issue for all models
        # for statement in all_statements["baseline"]["responses"]:
        #     annotation_items.append(("baseline", statement))
        for axis in ["econ", "govt", "scty", "dipl"]:
            for sign in ["+", "-"]:
                statements = all_statements[axis + sign]["responses"]
                for statement in random.Random(id).sample(statements, 5):
                    annotation_items.append((id, axis + sign, statement.replace("\n", "")))
                    id += 1
    with open(os.path.join(this_base_directory, f"outputs/8values/annotation_items_{model_type}.tsv"), "w", encoding="utf8") as f:
        r = csv.writer(f, delimiter="\t")
        r.writerows(annotation_items)
    return annotation_items

def annotator_scores_and_agreement(annotators=["anno1","anno2"], verbose=True):
    """
    Calculates and prints strict agreement, lenient agreement weighted Cohen's kappa, spearman's rho and ICC(A,1) for two annotators/LLM judges.
    Options are anno1, anno2, human_avg, llm, llm_allaxes.
    Human judgements are read from file, LLM judgements are calculated here (requires OpenAI API token to be set).
    If one annotator is human_avg, this function is recursively called on ["anno1","anno2"].
    """
    #40 items, 5 per axis/dir
    coherent_judgements = {annotator: [] for annotator in annotators}
    scores = {axis: copy.deepcopy({annotator: {} for annotator in annotators}) for axis in VALUES}
    for annotator in annotators:
        if annotator.startswith("llm"):
            judge = Model("gpt-5.4")
            with open(os.path.join(this_base_directory, f"outputs/8values/annotation_items.tsv")) as f:
                items = list(csv.reader(f, delimiter="\t"))
            with open(os.path.join(this_base_directory, f"outputs/8values/annotation_items_mistral-medium.tsv")) as f2:
                mistral_items = list(csv.reader(f2, delimiter="\t"))
                for item in mistral_items: item[0] = int(item[0]) + len(items) #shift by 40 to get unique IDs
                items.extend(mistral_items)
                #items.sort(key=lambda x:x[0]) #sort by id
                for item in items:
                    id = int(item[0])
                    statement = item[2]
                    for judging_axis in VALUES:
                        if annotator == "llm_allaxes":
                            prompt = construct_prompt_LLMjudge_allaxes(statement)
                            res = judge_statement(judge, prompt, score_type="list")
                            for a, axis in enumerate(VALUES): scores[axis][annotator][id] = res[a]
                            break
                        else:
                            prompt = construct_prompt_LLMjudge_old(statement, axis=VALUES[judging_axis]["name"], value_pos=VALUES[judging_axis]["positive"], value_neg=VALUES[judging_axis]["negative"])
                            scores[judging_axis][annotator][id] = judge_statement(judge, prompt)
            print(f"{judge.token_usage} tokens used, total cost: {judge.total_cost}")            
                        
        elif annotator == "human_avg":
            avgs = annotator_scores_and_agreement(verbose=True)
            for axis in VALUES:
                scores[axis][annotator] = avgs[axis]
        else:
            with open(os.path.join(this_base_directory, "RepresentationPoliticalLLM/annotations", f"annotations_{annotator}.tsv"), encoding="utf8") as f:
                for line in f:
                    id, _, econ_score, govt_score, scty_score, dipl_score, coherent = line.strip().split("\t")
                    scores["econ"][annotator][int(id)] = int(econ_score)
                    scores["govt"][annotator][int(id)] = int(govt_score)
                    scores["scty"][annotator][int(id)] = int(scty_score)
                    scores["dipl"][annotator][int(id)] = int(dipl_score)
                    coherent_judgements[annotator].append(coherent)
    
    averages = {}
    # strict agreement and lenient agreement (within 1)
    for axis, axis_scores in scores.items():
        sorted_anno1 = [s[1] for s in sorted(scores[axis][annotators[0]].items(), key=lambda x:x[0])]
        sorted_anno2 = [s[1] for s in sorted(scores[axis][annotators[1]].items(), key=lambda x:x[0])]
        assert len(sorted_anno1) == len(sorted_anno2)
        if verbose:
            print(sorted_anno1)
            print(sorted_anno2)
            anno_differences = np.array(sorted_anno1) - np.array(sorted_anno2)
            lenient_agreement = sum(abs(anno_differences)<=1)
            print(f"{axis}: Strict agreement {sum(anno_differences==0)/len(anno_differences)}, lenient agreement {lenient_agreement/len(anno_differences)}")
            print(f"Weighted Cohen's kappa: {cohen_kappa_score([round(s) for s in sorted_anno1], [round(s) for s in sorted_anno2], weights='quadratic')}")
            print(f"Spearman's rho: {spearmanr(sorted_anno1, sorted_anno2).statistic}, p={spearmanr(sorted_anno1, sorted_anno2).pvalue}")
            df = pd.DataFrame( {
                "item": list(range(len(sorted_anno1))) * 2,
                "rater": ["Rater_1"] * len(sorted_anno1) + ["Rater_2"] * len(sorted_anno2),
                "score": sorted_anno1 + sorted_anno2})

            icc_table = pingouin.intraclass_corr(data=df, targets="item", raters="rater", ratings="score")
            print("ICC(A,1)", icc_table.loc[1]["ICC"]) #ICC(A,1) is what Kim et al. reported
        if coherent_judgements[annotators[0]] and coherent_judgements[annotators[1]]:
            print(cohen_kappa_score(coherent_judgements[annotators[0]], coherent_judgements[annotators[1]]))
        averages[axis] = {i: (axis_scores[annotators[0]][i] + axis_scores[annotators[1]][i])/ 2 for i in range(len(axis_scores[annotators[0]]))}
    return averages


if __name__ == "__main__":
    #annotator_scores_and_agreement()#["human_avg", "llm_allaxes"])
    # measure_entanglement("llama-big", "_gpt5")
    # raise Exception("done")

    parser = argparse.ArgumentParser(description="Multi-axis political steering")
    parser.add_argument("--model_type", type=str, default="llama-big", help="e.g. llama, llama-big")
    parser.add_argument("--task", type=str, default="get_axis_predictions")
    parser.add_argument("--category", type=str, default="politician_us", help="Only for get_axis_predictions, e.g. politician_us")
    parser.add_argument("--prompt", type=str, default="values", help="Only for generate_8values_statements, e.g. values, issues")
    parser.add_argument("--strength", type=float, default=.5)
    parser.add_argument("--all8", type=bool, default=False)
    parser.add_argument("--orth", default=False)
    parser.add_argument("--baseline", type=str, default="axes_combined", help="Only for projection: baseline dataset to calculate axis winrate against")
    args = parser.parse_args()
    print(f"Running {args.task} on {args.model_type}")

    if args.model_type in ["llama", "qwen"]:
        layer_ids = range(9, 31)
    else:
        layer_ids = None
    if args.task in ["projection", "robustness_prompt", "get_axis_predictions", "evaluate_8values_steering", "generate_8values_statements", "steer_multiaxis", "congress_groups"]:
        model = Model_with_steering(args.model_type, steering_str="", dataset="", strength=args.strength, layer_ids=layer_ids)

    #### Tasks without model
    if args.task == "judge_train_data":
        judge_train_data()
    elif args.task == "judge_all_generations":
        judge_all_generations(args.model_type, orthogonal=args.orth, all8=args.all8, by_axis=False, judge_all_prompt=True, ratio_all8=1)
    elif args.task == "annotator_scores_and_agreement":
        annotator_scores_and_agreement(["human_avg", "llm_allaxes"])
    elif args.task == "judge_combinations":
        judge_combinations(args.model_type, orthogonal=args.orth, all8=args.all8)
    elif args.task == "measure_entanglement":
        measure_entanglement(args.model_type, suffix="_gpt5")
    elif args.task == "evaluate_multiaxis_steering":
        evaluate_multiaxis_steering(args.model_type)

    ### Tasks with model
    if args.task == "projection":
        projection(model, args.model_type, args.all8, args.orth, baseline=args.baseline, ratio_all8=1)

    elif args.task == "robustness_prompt":
        prompt_variants = ["Generate a 2026 statement by the {} party {}.",
        "Create a statement by {} party {} on a current political issue.",
        "Impersonate a politician of {} party {} and give a statement.",
        "Generate political opinions from the perspective of {} party {}.",
        "Give me a press release that {} party {} might put out."]
        countries = ["Germany", "Italy", "France", "United Kingdom", "Sweden"]
        all_parties = [p for country in countries for p in country_to_parties[country]]
        manifesto_scores = read_manifesto_scores(countries, all_parties, columns="all")
        num_variables = len(manifesto_scores["SPD"])
        prompt_results = {}
        by_variable = {var:[] for var in manifesto_scores["SPD"]}
        for i, prompt_template in enumerate(prompt_variants):
            setup = "-".join([model.model_type, str(args.all8), str(args.orth), f"prompt{i}"])
            res_dict = calculate_party_scores(model, setup, countries, prompt_template=prompt_template)
            #evaluate_scaling(res_dict["results"], manifesto_scores)
            winrates, var_wins = scaling_winrate(manifesto_scores, res_dict, args.baseline)
            print(winrates)
            for axis in var_wins:
                for var in var_wins[axis]: by_variable[var].append(axis)
            prompt_results[i] = winrates
        for axis in prompt_results[0].keys():
            avg = np.mean([prompt_results[i][axis] for i in prompt_results])
            sd = np.std([prompt_results[i][axis] for i in prompt_results])
            print(f"{axis}: Average winrate {avg}, sd {sd}")
        variance_by_variable = {i: 0 for i in range(1,6)}
        for var, axes in by_variable.items():
            variance_by_variable[len(set(axes))] += 1
        print(variance_by_variable)
        print(prompt_results)
        
    elif args.task == "get_axis_predictions":
        vectors = get_axes_vectors(datasets=get_axes_datasets(all8=args.all8), model=model, method="rfm")
        get_axis_predictions(model, args.category, vectors, strength=args.strength, orthogonal=args.orth, all8=args.all8)
    elif args.task == "evaluate_8values_steering":
        evaluate_8values_steering(model, strength=args.strength, all_combos=False)
    elif args.task == "generate_8values_statements":
        generate_8values_statements(model, args.strength, prompt=args.prompt, all8=args.all8, orthogonal=args.orth, ratio_all8=1)
    elif args.task == "steer_multiaxis":
        model.vectors = get_axes_vectors(datasets=get_axes_datasets(all8=args.all8), model=model, method="rfm")
        steer_multiaxis(model, all8=args.all8)
    elif args.task == "congress_groups":
        group_vectors = get_all_angles(args.model_type, method="rfm")
        lml_vec = [vector for vector in group_vectors if vector.name == "left-modleft"][0]
        alll_allr_vec = [vector for vector in group_vectors if vector.name == "main"][0]
        axes_vectors = get_axes_vectors(datasets=get_axes_datasets(all8=args.all8), model=model, method="rfm")
        print(len(axes_vectors))
        for axis_vec in axes_vectors:
            print(axis_vec.name, "lml", 2-avg_angle(lml_vec, axis_vec))
        for axis_vec in axes_vectors:
            print(axis_vec.name, "main", 2-avg_angle(alll_allr_vec, axis_vec))


