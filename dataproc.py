from directory_config import *

import csv, os, random, json, numpy
import pandas as pd
from datasets import load_dataset
import datasets
datasets.utils.logging.disable_progress_bar()
from tqdm import tqdm

def optional_tqdm(iterable, do_tqdm=do_tqdm, **kwargs):
    if do_tqdm:
        return tqdm(iterable, **kwargs)
    else:
        return iterable

# issues from Wojatzki et al 2018, https://aclanthology.org/L18-1224.pdf (everything from BLM) plus some own additions
issues = ["politics", "the state of the US", "Abortion", "President Trump", "Antifa", "Black Lives Matter", "Climate Change", "Creationism in school", "Foreign Aid", "Gender Equality", "Gun Rights", "Marijuana", "Same-sex Marriage", "Mandatory Vaccination", "Media Bias", "Obama Care", "US Electoral System", "US in the Middle East", "US Immigration", "Vegetarianism & Veganism", "War on Terrorism"]

issues_by_axis = {
        "econ": ["socialism", "capitalism", "a planned economy", "free markets", "economic growth", "taxes", "welfare"],
        "govt": ["legalizing marijuana", "press freedom", "civil liberties", "authority", "policing", "surveillance", "social media regulation"],
        "scty": ["traditional values", "religion", "same-sex marriage", "LGBTQ+ rights", "creationism in school", "GMOs", "multiculturalism"],
        "dipl": ["US in the Middle East", "military funding", "pacifism", "globalism", "foreign aid", "the EU"]
    }

######### STEERING ######################
def get_political_dataset(dataset="hs116", prefixing=False, issues=False, opposing=False, length=3000):
    # issues from Wojatzki et al 2018, https://aclanthology.org/L18-1224.pdf (everything from BLM) plus some own additions
    baseline_prefixes = ["Here is a statement by a conservative Republican: As a conservative Republican, here is my opinion: ", "Here is a statement by a progressive Democrat: As a progressive Democrat, here is my opinion: "]
    if dataset == "hs116" or dataset == "politicians":
        left, _, right =  get_politician_statements(balanced=False, issues=issues)
    elif dataset == "politune":
        if prefixing == "prompts":
            left, _, right = get_politune_statements(include_prompts=True)
            if opposing == "prompts":
                left = [item[0] for item in left]
        elif prefixing == "baseline":
            left, _, right = get_politune_statements(prefixes=baseline_prefixes)
        else:
            left, _, right = get_politune_statements()
    elif dataset == "baseline":
        if issues:
            left, _, right = get_topic_statements(issues=issues)
        else:
            left, _, right = [baseline_prefixes[1]], None, [baseline_prefixes[0]]
    elif dataset == "generations":
        left, _, right = get_generated_statements(prefixing=prefixing, baseline_prefixes=baseline_prefixes, length=length)
    elif dataset == "nayak":
        left, _, right = get_nayak_statements()
    elif dataset == "all":
        l1, _, r1 = get_politician_statements(balanced=False, issues=issues)
        r1 = r1[:len(l1)]
        l2, _, r2 = get_politune_statements()
        r2 = r2[:len(l2)]
        l3, _, r3 = get_generated_statements(prefixing="baseline", baseline_prefixes=baseline_prefixes, length=length)
        left = l1 + l2 + l3
        right = r1 + r2 + r3
    elif dataset == "axes_combined":
        return get_axes_datasets(combined=True)
    else:
        raise Exception(f"Unknown dataset '{dataset}' passed to get_political_dataset")

    return [{"name": dataset, "positive": right, "negative": left}]

def get_nayak_statements():
    left_data = []
    right_data = []
    dataset = load_dataset("JyotiNayak/political_ideologies", split="train")
    for i in range(len(dataset)):
        example = dataset[i]
        text = example['statement']
        if example['label'] == 0:
            right_data.append([text])
        elif example['label'] == 1:
            left_data.append([text])
    return left_data, None, right_data 

def get_generated_statements(prefixing, baseline_prefixes, length):
    left = []
    right = []
    with open(os.path.join(this_base_directory, "outputs/essays", f"biased_essays_Llama-3.1-8B-Instruct_left.tsv")) as f:
        r = csv.reader(f, delimiter="\t")
        _ = next(r)
        for line in r:
            if prefixing == "baseline":
                left.append([baseline_prefixes[1] + line[0][:length]])
            else:
                left.append([line[0][:length]])
    with open(os.path.join(this_base_directory, "outputs/essays", f"biased_essays_Llama-3.1-8B-Instruct_right.tsv")) as f:
        #self.model_string.split("_")[0].split()[0]
        r = csv.reader(f, delimiter="\t")
        _ = next(r)
        for line in r:
            if prefixing == "baseline":
                right.append([baseline_prefixes[0] + line[0][:length]])
            else:
                right.append([line[0][:length]])
    return left, None, right

def load_dw_nominate(congress="HS116"):
    with open(this_base_directory + f'RepresentationPoliticalLLM/politician_data/{congress}_members.csv', "r") as f:
        delimiter = "," if congress == "HS118" else "\t"
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader)
        #print(header)
        all_scores = []
        for row in reader:
            if row[13]:
                all_scores.append({"bioname": row[9], "nominate_dim1": float(row[13])})
    df = pd.DataFrame.from_records(all_scores)
    name_ds = load_dataset("erickrribeiro/gender-by-name", streaming=True)
    name_dict = {}
    for row in iter(name_ds["train"]):
        if row["Count"] > 1000 and (row["Name"] not in name_dict or name_dict[row["Name"]][1] < row["Count"]):
            name_dict[row["Name"]] = (["female", "male"][row["Gender"]], row["Count"])
    for row in iter(name_ds["test"]):
        if row["Count"] > 1000 and (row["Name"] not in name_dict or name_dict[row["Name"]][1] < row["Count"]):
            name_dict[row["Name"]] = (["female", "male"][row["Gender"]], row["Count"])
    #print("debug: loaded name dict of size", len(name_dict))
    df['first_middle_name'] = df['bioname'].apply(lambda x: x.split(', ')[1])
    df['first_name'] = df['first_middle_name'].apply(lambda x: x.split(' ')[0] + ' ' if ' ' in x else x + ' ')
    df['middle_name'] = df['first_middle_name'].apply(lambda x: x.split(' ')[1][0] + '. ' if ' ' in x else '')
    df['last_name'] = df['bioname'].apply(lambda x: x.split(', ')[0].lower().capitalize().strip())
    df['fullname'] = df['first_name'] + df['middle_name'] + df['last_name']
    for index, row in df.iterrows():
        if row['nominate_dim1'] > 1 or row['nominate_dim1'] < -1:
            row['nominate_dim1'] /= 1000
        assert row['nominate_dim1'] < 1 and row['nominate_dim1'] > -1
    return df, name_dict

def load_voteview_vote(votename:str="fisa"):
    member_names = []
    vote_labels = [] #0 = Nay, 1 = Yea
    with open(os.path.join(this_base_directory, "RepresentationPoliticalLLM/politician_data", f"{votename}_vote.json")) as f:
        data = json.load(f)
        votes = data["rollcalls"][0]["votes"]
        for member in votes:
            if member["vote"] in ["Yea", "Nay"]:
                try:
                    vote_labels.append({"Yea": 1, "Nay": 0}[member["vote"]])
                    if "first_name" in member: #Senate
                        member_names.append(member["first_name"] + " " + member["last_name"].lower())
                    else: #House makes it more difficult
                        last, first = member["name"].lower().split(",")[:2]
                        member_names.append(first.split()[0] + " " + last) #re.sub(r"\(.*\)|\".*\"", "", first)
                except Exception as e:
                    print(e)
                    print(member)
    assert len(member_names) == len(vote_labels)
    print("debug", len(vote_labels))
    return member_names, vote_labels

def get_politician_statements(return_list=False, balanced=True, only_gender=False, issues=[]):
    """
    * if return_list is False, returns left_statements, centrist_statements, right_statements as lists of 2-element lists, representing user-assistant conversation turns.
    * if return_list is True, returns all statements as list of (dw_score, turns list as above)
    * balanced: both left and right statements are balanced by gender (though not necessarily of the same length). if return_list is True, this is ignored
    * only_gender should be one of "male", "female" or something falsy, in the latter case all congresspeople are included
    * issues is a list of issues, e.g. ["abortion", "gun control", ...]. Recommended only for use with balanced=True
    """
    df, name_dict = load_dw_nominate()
    statements = []
    statements_by_gender = {"male": [], "female": []}
    
    for index, row in df.iterrows():
        dw_score = row['nominate_dim1']
        if row['first_name'].strip() in name_dict:
            statements_by_gender[name_dict[row['first_name'].strip()][0]].append((dw_score, [f"Generate a statement by {row['fullname']}, a politician in the United States.", f"In 2019, {row['fullname']} said that"]))
        if not only_gender or (row['first_name'].strip() in name_dict and name_dict[row['first_name'].strip()][0] == only_gender):
            if issues:
                for issue in issues:
                    statements.append((dw_score, [f"Generate a statement on {issue} by {row['fullname']}, a politician in the United States.", f"In 2019, {row['fullname']} said on {issue} that"]))
            else:
                statements.append((dw_score, [f"Generate a statement by {row['fullname']}, a politician in the United States.", f"In 2019, {row['fullname']} said that"]))
    if balanced:
        left_f = [s[1] for s in statements_by_gender["female"] if s[0]<0]
        left_m = [s[1] for s in statements_by_gender["male"] if s[0]<0]
        right_f = [s[1] for s in statements_by_gender["female"] if s[0]>0]
        right_m = [s[1] for s in statements_by_gender["male"] if s[0]>0]
        right_balanced = right_f + right_m[:len(right_f)]
        random.Random(15).shuffle(right_balanced)
        left_balanced = left_f + left_m[:len(left_f)]
        assert len(left_balanced) == 2 * len(left_f)
        random.Random(15).shuffle(left_balanced)
        #careful: this DOES NOT guarantee that left and right have the same length, just that both are balanced

    statements.sort(key=lambda x: x[0]) # sort by political leaning
    left_statements = [s[1] for s in statements if s[0]<0][:200]
    right_statements = [s[1] for s in statements if s[0]>0][-200:] #might be less than 200 if e.g. there aren't enough right-wing women
    centrist_statements = [s[1] for s in statements[int(len(statements)/2)-20: int(len(statements)/2)+20]]
    right_statements.reverse() # this will pair up moderates with moderates, radical with radical

    if return_list:
        return statements
    elif balanced:
        return left_balanced, None, right_balanced
    else:
        return left_statements, centrist_statements, right_statements

def get_politune_statements(include_prompts=False, prefixes=[], length=10000):
    """
    uses politune-right preference set. There is also a left one, idk if that's the same statements swapped
    prefixes, if provided, should be a list of two strings representing the left and right prefix in that order
    """
    #left_data = load_dataset("scale-lab/politune-left", split="train") 
    all_data = load_dataset("scale-lab/politune-right", split="train")
    right_data = []
    left_data = []
    for row in all_data:
        if include_prompts:
            right_data.append([row["prompt"], row["chosen"][:length]])
            left_data.append([row["prompt"], row["rejected"][:length]])
        elif prefixes:
            right_data.append([prefixes[0] + row["chosen"][:length]])
            left_data.append([prefixes[1] + row["rejected"][:length]])
        else:
            right_data.append([row["chosen"][:length]])
            left_data.append([row["rejected"][:length]])
    #right_data = all_data.map(format_instruction_template_right, remove_columns=['prompt', 'chosen', 'rejected'], num_proc=4)
    #left_data = all_data.map(format_instruction_template_left, remove_columns=['prompt', 'chosen', 'rejected'], num_proc=4)
    print(f"{len(all_data)} train instances")
    
    return left_data, None, right_data #list(left_data["messages"]), None, list(right_data["messages"])

def get_topic_statements(right_str="conservative Republican", left_str="progressive Democrat", issues=issues):
    left_statements, right_statements = [], []
    for issue in issues:
        right_statements.append([f"Here is a statement by a {right_str}: As a {right_str}, my opinion on {issue} is that"])
        left_statements.append([f"Here is a statement by a {left_str}: As a {left_str}, my opinion on {issue} is that"])
    return left_statements, None, right_statements

def get_8values_questions():
    with open(this_base_directory + "RepresentationPoliticalLLM/8values.json") as f:
        questions = json.load(f)
        questions_by_axis = {axis: {"positive": [], "negative": []} for axis in questions[0]["effect"].keys()}
        for question in questions:
            for axis in question["effect"].keys():
                if question["effect"][axis] == 10:
                    questions_by_axis[axis]["positive"].append(question["question"])
                elif question["effect"][axis] == -10:
                    questions_by_axis[axis]["negative"].append(question["question"])
        print("Total statements by axis: ", {axis: len(statements["positive"]) + len(statements["negative"]) for axis, statements in questions_by_axis.items()})
        return questions_by_axis

def get_axes_datasets(all8=False, combined=False, ratio_all8=.5):
    """
    order: econ, govt, scty, dipl
    all8 outputs individual datasets for all values (2 per axis), it adds ~30% positive examples from other axes to negative class in order to separate dimensions
    """
    if not ratio_all8: ratio_all8 = 1
    with open(this_base_directory + "RepresentationPoliticalLLM/8values_generated_statements.json", "r") as f:
        statements_by_axis = json.load(f)
        for axis in statements_by_axis:
            axis["positive"] = [[s] for s in axis["positive"]] #put in correct format
            axis["negative"] = [[s] for s in axis["negative"]] #put in correct format
        
        if all8:
            statements_8_axes = []
            for axis in statements_by_axis:
                ds_pos = {"name": axis["name"] + "_pos", "positive": axis["positive"].copy(), "negative": axis["negative"].copy()}
                ds_neg = {"name": axis["name"] + "_neg", "positive": axis["negative"].copy(), "negative": axis["positive"].copy()}
                len_ds_neg = len(ds_pos["negative"])
                for other_axis in statements_by_axis:
                    if other_axis["name"] != axis["name"]:
                        ds_pos["negative"].extend(random.Random(15).sample(other_axis["positive"], k=int(len_ds_neg*ratio_all8/3)))
                        ds_neg["negative"].extend(random.Random(15).sample(other_axis["negative"], k=int(len_ds_neg*ratio_all8/3)))
                random.Random(15).shuffle(ds_pos["negative"])
                random.Random(15).shuffle(ds_neg["negative"])
                statements_8_axes.append(ds_pos)
                statements_8_axes.append(ds_neg)
            return statements_8_axes  

    if combined:
        all_pos = []
        all_neg = []
        for ds in statements_by_axis:
            all_pos.extend(ds["positive"])
            all_neg.extend(ds["negative"])
        comb_ds = [{"name": "axes_combined", "positive": all_pos, "negative": all_neg}]  
        return comb_ds      

    return statements_by_axis

def read_manifesto_scores(countries:list[str], parties:list[str], columns=["rile","planeco","markeco","welfare","intpeace", "per104", "per107", "per108", "per201", "per501", "per503"]):
    """
    Output format: dict {party1: {var1: score, var2: score, ...}, ...}
    Gets most recent rows for specified parties of specified country (column "countryname") from csv,
    returns party name and specified columns for them. If columns == "all", returns all columns where at most 1 party (if single country) or 10% of parties (if multiple countries) have zero values (so scaling still works)
    """
    with open(os.path.join(this_base_directory, "RepresentationPoliticalLLM/politician_data", "MPDataset_MPDS2025a.csv")) as f:
        df = pd.read_csv(f)
        df_country = df[df["countryname"].isin(countries)]
        df_country = df_country[(df_country["partyname"].isin(parties)) | (df_country["partyabbrev"].isin(parties))]
        result = {}
        for _, row in df_country.iterrows():
            party_name = row["partyname"] if row["partyname"] in parties else row["partyabbrev"]
            if columns == "all":
                result[party_name] = {col: row[col] for col in row.index if col.startswith("per") and col not in ["pervote", "peruncod"]}
            else:
                result[party_name] = {col: row[col] for col in columns}
        if True: #columns == "all":
            cols_to_remove = []
            all_columns = result[parties[0]].keys()
            for column in all_columns:
                if len(countries) == 1: #Germany-only setting
                    if [result[party][column] for party in result].count(0) > 1 or any([numpy.isnan(result[party][column]) for party in result]):
                        cols_to_remove.append(column)
                else:   
                    if [result[party][column] for party in result].count(0) > (.2*len(parties)) or any([numpy.isnan(result[party][column]) for party in result]):
                        cols_to_remove.append(column)
            for party in result:
                result[party] = {k: v for k, v in result[party].items() if k not in cols_to_remove}
            #print(cols_to_remove)
            print(f"Removed {len(cols_to_remove)} columns, {len(result[parties[0]])} left")
        print(f"extracted scores for {len(result)} parties: ", list(result.keys()))
        print(f"{len(parties)-len(result)} parties not found: ", set(parties).difference(set(result.keys())))
        return result


###################### FILE MANAGEMENT AND LOGGING #########################

class CompactListEncoder(json.JSONEncoder):
    """A JSON Encoder that keeps lists of numbers/primitives on a single line."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.indentation_level = 0

    @property
    def current_indent(self):
        return " " * (self.indentation_level * (self.indent or 4))

    def encode(self, o):
        # If it's a list and only contains primitives (ints, floats, strings, bools, None)
        if isinstance(o, (list, tuple)):
            if all(not isinstance(x, (dict, tuple)) for x in o):
                # Format the primitive list cleanly in one line
                items = ", ".join(json.dumps(x) for x in o)
                return f"[{items}]"
            else:
                # Nested list handling
                self.indentation_level += 1
                lines = [f"{self.current_indent}{self.encode(item)}" for item in o]
                self.indentation_level -= 1
                return "[\n" + ",\n".join(lines) + f"\n{self.current_indent}]"
        
        elif isinstance(o, dict):
            if not o:
                return "{}"
            self.indentation_level += 1
            item_indent = self.current_indent
            lines = []
            for k, v in o.items():
                # Format keys safely as JSON strings, then recursively encode values
                key_str = json.dumps(k) 
                lines.append(f"{item_indent}{key_str}: {self.encode(v)}")
            self.indentation_level -= 1
            closing_indent = self.current_indent
            return "{\n" + ",\n".join(lines) + f"\n{closing_indent}" + "}"
            
        # 3. Fallback to basic types (numbers, strings, bools, None) safely
        # Using super().encode ensures we don't trigger infinite recursion
        return json.dumps(o) 
    
    def iterencode(self, o, _one_shot=False):
        # Overriding iterencode forces json.dump() to use our custom encode() logic
        yield self.encode(o)
            
