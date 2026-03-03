import copy
import json
import random
import re
from openai import OpenAI
import time
import nltk
from nltk.tokenize import sent_tokenize
import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

client = OpenAI(
    api_key="",
    base_url=""
)
save_history_dir='synthetic_data/history/'
metadir='meta/'
savedor='synthetic_data/'
relation_prompt=metadir+'relation_prompt-CDR.json'

print("openai.api_key: ",client.api_key)
print("relation_prompt: ",relation_prompt)
print("save_history_dir: ",save_history_dir)

def reconstruct_text(sents):
    if not sents: return ""
    return " ".join([" ".join(sent) for sent in sents])

@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type((Exception,)),
    reraise=False
)
def get_completion_from_messages(messages, model="", temperature=0):

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=60
        )
        content = response.choices[0].message.content.strip()
        if not content:
            raise ValueError("模型返回空内容")
        return content
    except Exception as e:
        if hasattr(e, 'status_code') and e.status_code == 429:
            retry_after = int(e.headers.get('Retry-After', 5)) if hasattr(e, 'headers') else 5
            print(f"⚠️ 429频率限制！将等待{retry_after}秒后重试...")
            time.sleep(retry_after)
        error_msg = str(e)[:80] + "..." if len(str(e)) > 80 else str(e)
        print(f"API调用错误: {error_msg}")
        raise


def generate(relation_types, train_triplets):
    ORIGINAL_CDR_PATH = "F:/CDR_json格式/convert_train.json"
    reference_str = ""
    try:
        with open(ORIGINAL_CDR_PATH, 'r', encoding='utf-8') as f:
            original_cdr = json.load(f)
        high_quality_original = [
            doc for doc in original_cdr
            if len(doc.get("vertexSet", [])) >= 5
               and len(doc.get("labels", [])) >= 6
               and any(label["r"] in [0, 1] for label in doc.get("labels", []))
        ]
        reference_samples = random.sample(high_quality_original, min(3, len(high_quality_original)))
        reference_str = ""
        for idx, ref in enumerate(reference_samples, 1):
            ref_context = " ".join([" ".join(sent) for sent in ref["sents"]]) if "sents" in ref else ref.get("context", "")
            ref_entities = []
            seen_entities = set()
            for entity_group in ref.get("vertexSet", []):
                if entity_group:
                    ent = entity_group[0]
                    ent_name = " ".join(ent['name'])
                    ent_key = f"{ent_name}|{ent['type']}"
                    if ent_key not in seen_entities:
                        seen_entities.add(ent_key)
                        ref_entities.append(f"- {ent_name} (type: {ent['type']})")
            ref_entities_str = "\n  ".join(ref_entities) if ref_entities else "No entities"
            ref_triplets = []
            rel_id_to_name = {0: "no chemical disease induction relation", 1: "chemical induced disease"}
            for label in ref.get("labels", []):
                h_ent = " ".join(ref["vertexSet"][label["h"]][0]["name"]) if (
                            "h" in label and ref["vertexSet"][label["h"]]) else "Unknown"
                t_ent = " ".join(ref["vertexSet"][label["t"]][0]["name"]) if (
                            "t" in label and ref["vertexSet"][label["t"]]) else "Unknown"
                rel_name = rel_id_to_name.get(label["r"], f"Unknown({label['r']})")
                ref_triplets.append(f"- ({h_ent}, {rel_name}, {t_ent})")
            ref_triplets_str = "\n  ".join(ref_triplets) if ref_triplets else "No triplets"
            reference_str += f"""
        Reference Sample {idx}:
        - Title: {ref.get("title", f"Original_CDR_{idx}")}
        - Clinical Context: {ref_context[:350]}...  # 截取临床相关内容
        - Key Entities (Chemical/Disease):
          {ref_entities_str}
        - Observed Relations (with clinical evidence):
          {ref_triplets_str}
        """
        print(f"Loaded {len(reference_samples)} high-quality CDR reference samples")
    except Exception as e:
        print(f"Error loading original CDR data: {e}")

    sampled_triplets = random.sample(train_triplets, min(5, len(train_triplets)))
    sampled_triplets_str = ""
    for idx, triplet in enumerate(sampled_triplets, 1):
        chem_list = triplet["chem"]
        dis_list = triplet["dis"]
        rel = triplet["rel"]
        raw_text = triplet.get("original_text", "No context.")
        chem_str = " ".join(chem_list)
        dis_str = " ".join(dis_list)
        chem_list_display = str(chem_list)
        dis_list_display = str(dis_list)
        if rel == "chemical induced disease":
            finding_desc = f"link {chem_list_display} (Chemical) to {dis_list_display} (Disease) — use ONLY THESE EXACT entities (no substitutes like 'estradiol valerate'). Include causal mechanism (e.g., 'via muscarinic receptor activation') and data (e.g., '12% incidence vs 3% control')."
        else:
            finding_desc = f"show NO association between {chem_list_display} (Chemical) and {dis_list_display} (Disease) — use ONLY THESE EXACT entities. Include negative evidence (e.g., 'p=0.45') and comparison (e.g., 'similar rates in exposed/unexposed')."
        sampled_triplets_str += f"""
             {idx}. MANDATORY Clinical Finding (NO entity replacement allowed):
             - Requirement: Must {finding_desc}
             - Background Info (Real Science)**: "{raw_text}"
             - Critical Rule: If you replace {chem_list_display} or {dis_list_display} with other entities, this response is INVALID."""

    for key_relation in relation_types:
        print(f"\n=== Generating for relation: {key_relation} ===")
        relationStr = "\", \"".join(relation_types[key_relation])
        messages = [
            {'role': 'system',
             'content': 'You are a clinical pharmacologist specializing in drug safety. Generate authentic PubMed-style abstracts that integrate drug-disease relations into cohesive clinical stories (no templates, no rigid sections). Entities MUST be word lists (e.g., ["major", "depression"]) for alignment with original CDR data.'}
        ]
        prompts = [
            f"""Standardize the provided raw biomedical triplets into a set of valid triplets (T_clear).

        **Task**:
        1. Extract the head entity (h) and tail entity (t) from the raw data below, and ensure they align with international nomenclature standards (like MeSH).
        2. Verify the authenticity and reliability of the relation (r) based on the clinical evidence within the original background document (d).
        3. Use exact word lists for entities (e.g., ["oral", "contraceptives"]).

        **Raw Triplets and Original Document (d)**:
        {[f"Triplet {i + 1}: Chemical={str(t['chem'])}, Disease={str(t['dis'])}, Relation={t['rel']}. Clinical Evidence (d): {t.get('original_text', '')}" for i, t in enumerate(sampled_triplets)]}

        **Format Requirement**: Output ONLY a raw JSON array of standardized triplets. NO extra text.
        [
          {{"h": ["oral", "contraceptives"], "t": ["oliguria"], "r": "no chemical disease induction relation"}}
        ]""",

            f"""Based on the standardized triplets (T_clear) you just verified, generate a fictional yet biomedically authentic document (d_syn).

        **Task**:
        1. Write a PubMed-style paragraph consisting of exactly 6-10 sentences.
        2. Semantically embed the entities (h and t) while maintaining the exact clinical context for their relation (r).
        3. Use the original clinical evidence provided in the previous step as demonstrations to ensure the linguistic style matches authentic clinical narratives.

        **Format Requirement**: Output ONLY a raw JSON object. NO Markdown, NO extra text.
        {{
          "title": "Concise Biomedical Title",
          "context": "The full generated PubMed-style paragraph without line breaks."
        }}""",

            f"""Construct the comprehensive synthetic sample (S_syn) based on your generated document (d_syn) and the standardized triplets (T_clear).

        **Task**:
        Integrate scenario-specific core fields into a unified JSON output that strictly complies with parsing specifications. For each relation, you MUST extract:
        1. The exact head (h) and tail (t) entities and their entity types (Chemical or Disease).
        2. The reasoning explanation (exp) for the relation (r) based on your generated text.
        3. The specific 0-based indices of the supporting sentences (idx_sent) where the relation is expressed.

        **Format Requirement**: Output ONLY a raw JSON array. NO Markdown, NO extra text.
        [
          {{
            "head entity": ["oral", "contraceptives"],
            "tail entity": ["oliguria"],
            "relation type": "no chemical disease induction relation",
            "reasoning explanation": "Explanation of how the text supports this relation...",
            "index of supporting sentence that shown in document": [1, 3]
          }}
        ]"""
        ]
        index = 0
        for pro in prompts:
            messages.append({'role': 'user', 'content': f"""{pro}"""})
            flag = 1
            thref = 0
            while flag:
                try:
                    if index == 0:
                        temperature = 1
                    else:
                        temperature = 0.0
                    response = get_completion_from_messages(messages, temperature=temperature)

                except Exception as e:
                    print(e)
                    if thref >= 3:
                        break
                    print("wait 2 seconds!")
                    thref += 1
                    time.sleep(2)
                else:
                    print("success!")
                    flag = 0
            if flag == 1:

                break
            messages.append({'role': 'assistant', 'content': f"""{response}"""})
            index += 1
        if index != len(prompts):
            print("skip this doc!")
            continue
        save_path = save_history_dir + 'history_' + key_relation + '-(固定头尾实体)' + '.json'
        try:
            history = json.load(open(save_path))
        except Exception as e:
            history = []
        history.append(messages)
        history = json.dumps(history)
        with open(save_path, 'w') as outfile:
            outfile.write(history)
        outfile.close()

def static():
    history=[]
    relation_types = json.load(open(relation_prompt))
    for key_relation in relation_types:
        try:
            load_path = save_history_dir + 'history_' + key_relation + '-(固定头尾实体)' + '.json'
            his = json.load(open(load_path))
        except Exception as e:
            print("no such file!")
            print(load_path)
            his = []
        for hi in range(0, len(his)):
            item = his[hi]
            temp = {'role': 'tag'}
            temp['content'] = key_relation
            item.append(temp)
            history.append(item)

    train_data_path = "F:/CDR_json格式/convert_train.json"
    train_triplets = extract_train_triplets(train_data_path)
    train_chem_set = set(["_".join(t["chem"]).lower() for t in train_triplets])
    train_dis_set = set(["_".join(t["dis"]).lower() for t in train_triplets])
    print("Number of origin synthetic data in current seed: ",len(history))
    dataset=[]
    index=0
    for message in history:
        print("index: ", index)
        onedata={}
        try:
            doc=message[2]['content']
            doc=doc.replace("\n","")
            doc=doc.replace("\"Hakuna Matata.\"", "\\\"Hakuna Matata\\\"")
            doc = doc.replace("Here is a fictional paragraph that describes the relation types \"place of birth\", \"place of death\", \"father\", \"mother\", and \"position held\":", "")
            doc = doc.replace("Here's an example paragraph:", "")
            doc = doc.replace("```{", "{")
            doc = doc.replace("```This paragraph contains relations \"place of birth\" (Pella), \"father\" (King Philip II of Macedonia), \"mother\" (Queen Olympia), \"position held\" (King of Macedonia), \"place of death\" (Babylon).", "")
            doc = doc.replace("JSON Format:", "")
            doc=doc.replace("Relation Types:- Place of birth: \"Stagira, Greece\"- Father: \"Nicomachus\"- Mother: \"Phaestis\"- Place of death: \"Euboea, Greece\"- Position held: \"Philosopher and scientist\"", "")
            doc=doc.replace("As an information extraction assistant, I have generated a fictional paragraph with the following relation types: \"place of birth,\" \"place of death,\" \"father,\" \"mother,\" and \"position held.","")
            doc=doc.replace("\"New Beginnings,\"","\\\"New Beginnings\\\",")
            doc = doc.replace("\"In Memory Of,\"", "\\\"In Memory Of\\\",")
            doc = doc.replace("\"The Lost Souls\"", "\\\"The Lost Souls\\\"")
            doc=doc.replace("\"{  \"title\"","{  \"title\"")
            doc=doc.replace("\"Shallow\"","\\\"Shallow\\\"")
            doc=doc.replace("```json{","{")
            doc=doc.replace("JSON format:{","{")
            doc=doc.replace("}```","}")
            doc=json.loads(doc)
        except Exception as e:
            print("*********** without content ***********")
            print(f"*********** 异常发生 at index {index} ***********")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            print(f"当前处理的 content: {doc}")
            print("错误位置附近内容:", doc[810:811])
            index += 1
            continue
        try:
            onedata['title']=doc['title']
            onedata['sents']=doc['context']
        except Exception as e:
            print("*********** without title or content ***********")
            index += 1
            continue

        try:
            entities=message[4]['content']
            entities=entities.replace("\n","")
            entities = entities.replace("},]", "}]")
            entities = entities.replace("\"}    {\"", "}, {")
            entities = entities.replace("{entity", "{\"entity")
            entities = entities.replace("\"Miscellaneous}", "\"Miscellaneous\"}")
            entities = entities.replace("JSON Format:", "")
            entities = entities.replace("```[", "[")
            entities = entities.replace("```The identified entity types are:- Person: Alexander the Great, Philip II of Macedonia, Queen Olympia, Aristotle- Location: Pella, Macedonia, Persian Empire, Hellenistic, Babylon- Time: July 356 BC, June 10, 323 BC- Number: 20, 32- Miscellaneous: Hellenistic, Greek", "")
            entities = entities.replace("Here are the extracted entities from the document:", "")
            entities=entities.replace("Here are the extracted entities in List of JSON format with the following keys: entity, entity type:","")
            entities=entities.replace("Note: The entity type for \"384 BC\" is \"Time\" and \"Philosopher and scientist\" is \"Miscellaneous\".","")
            entities = entities.replace("```json[", "[")
            entities = entities.replace("]```", "]")
            entities = json.loads(entities)
            for ent in entities:
                if isinstance(ent["entity"], str):
                    ent["entity"] = ent["entity"].strip().split()
            entities = [
                ent for ent in entities
                if ent["entity"] and ent["entity type"] in ["Chemical", "Disease"]
            ]
            generated_chems = []
            generated_dis = []
            for ent in entities:
                ent_list = ent["entity"]
                ent_key = "_".join(ent_list).lower()
                ent_type = ent["entity type"]
                if ent_type == "Chemical":
                    if ent_key not in train_chem_set:
                        print(f"❌ 生成无效Chemical实体：{ent_list}（不在训练集三元组中）")
                        entities.remove(ent)
                    else:
                        generated_chems.append(ent_key)
                elif ent_type == "Disease":
                    if ent_key not in train_dis_set:
                        print(f"❌ 生成无效Disease实体：{ent_list}（不在训练集三元组中）")
                        entities.remove(ent)
                    else:
                        generated_dis.append(ent_key)
        except Exception as e:
            print("*********** entity ***********")
            print(f"*********** 异常发生 at index {index} ***********")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            print(f"当前处理的 content: {entities}")
            index += 1
            continue
        vertexSet=[]
        for entity in entities:
            vertex={}
            try:
                vertex['name']=entity['entity']
                if 'entity type' in entity.keys():
                    vertex['type']=entity['entity type']
                elif 'entity_type' in entity.keys():
                    vertex['type']=entity['entity_type']
                else:
                    print("*********** vertex keys ***********")
            except Exception as e:
                print("*********** vertex part ***********")
                index += 1
                continue
            vertexSet.append(vertex)
        onedata['vertexSet']=vertexSet
        try:
            relations=message[12]['content']
            relations = relations.replace("\n", "")
            relations=relations.replace("Here's the triplet information organized in List of JSON format, as per your request:```","")
            relations = relations.replace("```Note: The remaining seven triples cannot be supported or explained by the provided context or they might be incorrect interpretations.", "")
            relations = relations.replace("\": None","\": \" \" ")
            relations = relations.replace("Sure, here it is::", "")
            relations = relations.replace("Here is the information organized in a JSON format with the requested keys for each relational triplet:```", "")
            relations = relations.replace("]```", "}]")
            relations = relations.replace("JSON Format:","")
            relations = relations.replace("```[", "[")
            relations= relations.replace("```For each triplet, the JSON object includes:- head entity: the main entity in the relationship- tail entity: the secondary entity in the relationship- relation type: the type of relationship between the entities- reasoning: an explanation of the significance of the relationship- context: the relevant sentence in which the relationship is mentioned in the original document.", "")
            relations = relations.replace("Sure, here is the information organized in a list of JSON format:```", "")
            relations = relations.replace("Sure, here it is:", "")
            relations=relations.replace("}}]","}]")
            relations=relations.replace("}}]For each triplet, the JSON object includes:- head entity: the main entity in the relationship- tail entity: the secondary entity in the relationship- relation type: the type of relationship between the entities- reasoning: an explanation of the significance of the relationship- context: the relevant sentence in which the relationship is mentioned in the original document.","}]")
            relations=relations.replace("Sure, here is the information organized in a list of JSON format:","")
            relations=relations.replace("Sure, here is the information organized in a list of JSON format with the requested keys:","")
            relations=relations.replace("Sure, here is the organized triplet information in List of JSON format:","")
            relations=relations.replace("This JSON format organizes the extracted relational triplets, along with their reasoning explanations and supporting sentences, in a structured and easy-to-read format.","")
            relations=relations.replace("Sure, here is the information organized in a list of JSON format with the requested keys:","")
            relations=relations.replace("I apologize for the confusion earlier, but I did not generate any relational triplets in the previous document. However, I can generate a list of JSON format for the entities extracted from the document with their corresponding support sentences. Here is the list:","")
            relations=relations.replace("I apologize for the confusion earlier. Here are the relational triplets based on the generated document, organized in a list of JSON format with the requested keys:","")
            relations=relations.replace("I apologize for the confusion earlier, as the document I generated did not contain any personal relationships or positions held by individuals. Therefore, I cannot provide reasoning explanations or supporting sentences for the extracted relational triplets.However, I can provide a general explanation of how the information can be organized in a list of JSON format. The list can contain one or more relational triplets, each represented as a JSON object with the following keys:- \"head entity\": the entity that appears as the subject of the relationship- \"tail entity\": the entity that appears as the object of the relationship- \"relation type\": the type of relationship between the head and tail entities- \"reasoning explanation\": a brief explanation of how the relationship was inferred from the text- \"supporting sentence\": the complete context of the sentence in the text that provides evidence for the relationshipHere is an example of how the information can be organized in a list of JSON format:","")
            relations=relations.replace("Sure, here is the organized information in List of JSON format:","")
            relations=relations.replace("Sure, here is the organized triplet information in a list of JSON format:","")
            relations=relations.replace("As mentioned earlier, I cannot generate relational triplets for the above document as it does not contain any information related to personal relationships or positions held by individuals. However, if there were such information, I could organize the triplet information in a list of JSON format with the following keys:","")
            relations=relations.replace("Note that this is just an example, and the actual JSON format may vary depending on the specific information and relationships extracted from the text.","")
            relations=relations.replace("Sure, here is the information organized in List of JSON format","")
            relations=relations.replace("I apologize for the confusion earlier. As mentioned earlier, there are no relational triplets in the generated document. However, I can provide a list of JSON format for each extracted entity in the document with the following keys:- Head entity- Tail entity- Relation type- Reasoning explanation of each relational triplet- Complete context of supporting sentence that shown in document","")
            relations=relations.replace("As there are no relational triplets in the generated document, I cannot provide the requested information. However, if there were relational triplets, the JSON format with the requested keys would look like this:","")
            relations=relations.replace("cinematic experience.\"}","cinematic experience.\"}]")
            relations=relations.replace("Since there were no relational triplets in the generated document, I will provide a list of JSON format for each extracted entity with the following keys: entity, entity type, and supporting sentence. Here is the list:","")
            relations=relations.replace("Sure, here is the list of JSON format with the requested keys:","")
            relations=relations.replace("As there are no relational triplets in the generated document, I cannot provide the requested information. However, I can provide a sample JSON format for a relational triplet:```","")
            relations=relations.replace("Sure, here's the information organized in a list of JSON format:","")
            relations=relations.replace("For each triplet, the JSON object includes:- head entity: the main entity in the relationship- tail entity: the secondary entity in the relationship- relation type: the type of relationship between the entities- reasoning: an explanation of the significance of the relationship- context: the relevant sentence in which the relationship is mentioned in the original document.","")
            relations=relations.replace(":[    {","[{")
            relations=relations.replace(":[  {","[{")
            relations=relations.replace("```json{","{")
            relations = relations.replace("```json[", "[")
            relations=json.loads(relations)
            for rel in relations:
                for key in ["head entity", "tail entity"]:
                    if key in rel and isinstance(rel[key], str):
                        rel[key] = rel[key].strip().split()
        except Exception as e:
            print("*********** relations part ***********")
            index += 1
            continue
        labels=[]
        if len(relations)<=1:
            print("*********** structure check ***********")
        for rela in relations:
            onerelation={}
            try:
                if "head_entity" in rela.keys():
                    onerelation['h']=rela["head_entity"]
                elif "head entity" in rela.keys():
                    onerelation['h']=rela["head entity"]
                else:
                    print("*********** h labels part ***********")
                    continue
                if "tail_entity" in rela.keys():
                    onerelation['t']=rela["tail_entity"]
                elif "tail entity" in rela.keys():
                    onerelation['t']=rela["tail entity"]
                else:
                    print("*********** t labels part ***********")
                    continue
                if "relation_type" in rela.keys():
                    onerelation['r']=rela["relation_type"]
                elif "relation type" in rela.keys():
                    onerelation['r']=rela["relation type"]
                else:
                    print("*********** r labels part ***********")
                    continue
                if "reasoning_explanation" in rela.keys():
                    onerelation['reasoning']=rela["reasoning_explanation"]
                elif "reasoning explanation" in rela.keys():
                    onerelation['reasoning']=rela["reasoning explanation"]
                elif "explanation" in rela.keys():
                    onerelation['reasoning']=rela["explanation"]
                elif "reasoning explannation" in rela.keys():
                    onerelation['reasoning']=rela["reasoning explannation"]
                elif "reasoning" in rela.keys():
                    onerelation['reasoning']=rela["reasoning"]
                elif "reasoning explanantion" in rela.keys():
                    onerelation['reasoning']=rela["reasoning explanantion"]
                else:
                    print("*********** reasoning_explanation labels part ***********")
                    onerelation['reasoning']=""
                if "supporting_sentence" in rela.keys():
                    onerelation['evidence']=rela["supporting_sentence"]
                elif "supporting sentence" in rela.keys():
                    onerelation['evidence']=rela["supporting sentence"]
                elif "complete context" in rela.keys():
                    onerelation['evidence']=rela["complete context"]
                elif "supporting sentence context" in rela.keys():
                    onerelation['evidence']=rela["supporting sentence context"]
                elif "complete context of supporting sentence" in rela.keys():
                    onerelation['evidence']=rela["complete context of supporting sentence"]
                elif "context" in rela.keys():
                    onerelation['evidence']=rela["context"]
                elif "complete context sentence" in rela.keys():
                    onerelation['evidence']=rela["complete context sentence"]
                elif "supporting context" in rela.keys():
                    onerelation['evidence']=rela["supporting context"]
                elif "supporting_context" in rela.keys():
                    onerelation['evidence']=rela["supporting_context"]
                elif "complete_supporting_sentence" in rela.keys():
                    onerelation['evidence']=rela["complete_supporting_sentence"]
                else:
                    print("*********** evidence labels part ***********")
                    onerelation['evidence']=""
            except Exception as e:
                print("*********** labels part ***********")
                continue
            if onerelation['t']!=None and onerelation['r']!=None and onerelation['h']!=None:
                labels.append(onerelation)
            else:
                print(" *********** None part ***********")

        if len(labels)!=0:
            onedata['labels']=labels
        else:
            print(" *********** labels is null ***********")
            continue
        onedata["relation_tag"]=message[13]['content']
        dataset.append(onedata)
        index+=1
    print(len(dataset))
    print(len(history))
    print("saving rate: " ,(len(dataset)/len(history))*100)
    filterpath=savedor+'CDR_FilterDataset.json'
    dataset=json.dumps(dataset)
    with open(filterpath,'w') as f1:
        f1.write(dataset)
    f1.close()
    print("filter path: ",filterpath)

def findMention(sents, ent_name, ent_type):
    mentions = []
    sent_id = 0
    if not isinstance(ent_name, list) or len(ent_name) == 0:
        print(f"⚠️ 空实体名称或非词列表（类型：{ent_type}），跳过匹配")
        return mentions
    ent_len = len(ent_name)
    ent_tokens_lower = [token.lower() for token in ent_name]  # 词列表转小写用于匹配
    for sent in sents:
        sent_tokens_lower = [token.lower() for token in sent]
        sent_len = len(sent_tokens_lower)
        if ent_len > sent_len:
            sent_id += 1
            continue
        for i in range(sent_len - ent_len + 1):
            if sent_tokens_lower[i:i+ent_len] == ent_tokens_lower:
                original_ent_name = sent[i:i+ent_len]
                start_pos = i
                end_pos = i + ent_len
                mentions.append({
                    'name': original_ent_name,
                    'sent_id': sent_id,
                    'type': ent_type,
                    'pos': [start_pos, end_pos]
                })
        sent_id += 1
    return mentions

def findSentence(sents,name):#
    evidences=[]
    sent_id=0
    for sent in sents:
        lowesent=[s.lower() for s in sent]
        lowername=[n.lower() for n in name]
        if lowername==lowesent:
            evidences.append(sent_id)
        sent_id+=1
    return evidences


def transfer():
    ORIGINAL_DATA_PATH = "F:/CDR_json格式/convert_train.json"
    original_titles = set()
    with open(ORIGINAL_DATA_PATH, 'r', encoding='utf-8') as f:
         original_data = json.load(f)
    for item in original_data:
        original_title = int(item["title"])
        original_titles.add(original_title)
    max_original_title = max(original_titles) if original_titles else 100000
    print(f"原始数据集最大title：{max_original_title}，新title从 {max_original_title + 1} 开始")
    origin=json.load(open(savedor+'CDR_FilterDataset.json'))
    datasets=[]
    nltk.download('punkt')
    index=0
    rel_info=json.load(open(metadir+'rel_info-CDR.json'))
    info2rel={}
    for rel_id_str, rel_name in rel_info.items():
        lower_rel_name = rel_name.lower()
        rel_id = int(rel_id_str)
        info2rel[lower_rel_name] = rel_id  # 正确映射：关系名称→整数ID
    error5=0
    relationfact=0
    entityNumber=0
    SameContext=[]

    for doc in origin:
        data={}
        context=doc['sents']
        new_title = max_original_title + 1 + len(datasets)
        data['title'] = new_title
        try:
            context=sent_tokenize(context)
        except Exception as e:
            print(" *********** Filter by error 1 ***********")
            index+=1
            continue
        if ' '.join(context) in SameContext:
            print(" *********** Same context 1.1 *********** ")
        else:
            SameContext.append(' '.join(context))
        sentences=[]
        for sent in context:
            sentlist=re.findall(r'\w+|[^\w\s]', sent)
            sentences.append(sentlist)
        data['sents']=sentences

        entityL = []
        for entity in doc["vertexSet"]:
            entity_str = " ".join(entity['name']) if isinstance(entity['name'], list) else str(entity['name'])
            entityL.append(entity_str)
        entities = []
        entityR = []
        for relaE in doc['labels']:
            h = relaE['h']
            t = relaE['t']
            h_str = " ".join(h) if isinstance(h, list) else str(h)
            t_str = " ".join(t) if isinstance(t, list) else str(t)
            if h_str not in entityR:
                entityR.append(h_str)
            if t_str not in entityR:
                entityR.append(t_str)
            if h_str not in entityL and h_str != '' and isinstance(h, list):
                entities.append({'name': h, 'type': "Chemical"})
            if t_str not in entityL and t_str != '' and isinstance(t, list):
                entities.append({'name': t, 'type': "Disease"})
        for relaE in doc["vertexSet"]:
            ent_str = " ".join(relaE['name']) if isinstance(relaE['name'], list) else str(relaE['name'])
            if ent_str in entityR:
                entities.append(relaE)

        vertexSet = []
        entity_idmap = {}
        for entity in entities:
            ent_name = entity["name"]  # 词列表
            if not isinstance(ent_name, list) or len(ent_name) == 0:
                continue
            ent_type = entity.get("type", "MISC")
            if ent_type not in ["Chemical", "Disease"]:
                continue
            ent_key = "_".join(ent_name).lower()
            vertex = findMention(sentences, ent_name, ent_type)
            if len(vertex) != 0:
                vertexSet.append(vertex)
                entity_idmap[ent_key] = len(vertexSet) - 1
            else:
                print(f"⚠️ 未找到实体 {' '.join(ent_name)} 的提及")
        data['vertexSet'] = vertexSet
        relations=doc['labels']
        label_relation=[]
        for relation in relations:
            relaItem={}
            try:
                relaName=relation['r']
                if type(relaName)==list:
                    for overlap_r in relaName:
                        overlap_relation=copy.deepcopy(relation)
                        overlap_relation['r']=overlap_r
                        relations.append(overlap_relation)
                    continue
                if relaName=="located in":
                    relaName="located in the administrative territorial entity"
                elif relaName=="directed":
                    relaName="director"
                elif relaName=="directing and co-writing":
                    relaName = "director"
                if relaName=="birthplace":
                    relaName="place of birth"
                if relaName=="famous work":
                    relaName="notable work"
                if relaName=="birth location":
                    relaName="place of birth"
                if relaName=="occupation":
                    relaName="position held"
                if relaName=="Occupation":
                    relaName="position held"
                if relaName=="date of publication":
                    relaName="publication date"
                relaItem['r'] = info2rel[relaName.lower()]
            except Exception as e:
                error5 += 1
                continue
            try:
                if isinstance(relation['h'], list):
                    head_key = "_".join(relation['h']).lower()
                    head_raw = " ".join(relation['h'])  # 用于日志
                else:
                    head_key = relation['h'].strip().lower()
                    head_raw = relation['h']
                if isinstance(relation['t'], list):
                    tail_key = "_".join(relation['t']).lower()
                    tail_raw = " ".join(relation['t'])
                else:
                    tail_key = relation['t'].strip().lower()
                    tail_raw = relation['t']
                relaItem['h'] = entity_idmap[head_key]
                relaItem['t'] = entity_idmap[tail_key]

            except Exception as e:
                error5 += 1
                print(f"❌ 赋值h/t/r失败: {str(e)}（h={head_raw}, t={tail_raw}）")
                continue
            if relaItem['h']==relaItem['t']:
                continue
            head_mentions = data['vertexSet'][relaItem['h']]  # Head实体的所有提及
            head_sent_ids = set()
            for mention in head_mentions:
                head_sent_ids.add(mention['sent_id'])

            tail_mentions = data['vertexSet'][relaItem['t']]
            tail_sent_ids = set()
            for mention in tail_mentions:
                tail_sent_ids.add(mention['sent_id'])
            if head_sent_ids & tail_sent_ids:
                relaItem['dist'] = "NON-CROSS"
            else:
                relaItem['dist'] = "CROSS"

            if relation['evidence']==[] or relation['evidence']=='':
                relaItem['evidence']=[]
            else:
                if type(relation['evidence'])==str:
                    evidence=re.findall(r'\w+|[^\w\s]', relation['evidence'])
                    relaItem['evidence'] = findSentence(sentences, evidence)
                else:
                    evilist=[]
                    for onevi in evidence:
                        onevi=re.findall(r'\w+|[^\w\s]', onevi)
                        evilist+=findSentence(sentences, onevi)
                    relaItem['evidence'] = evilist
            relaItem['reasoning'] = relation['reasoning']
            label_relation.append(relaItem)

        if len(label_relation)==0:
            print(" *********** No relational fact in this Doc 6 ***********")
            index += 1
            continue
        relationfact+=len(label_relation)
        data['labels']=label_relation
        entityNumber+=len(data['vertexSet'])
        unseen_reltype=list(json.load(open(metadir+'rel2id_unseen-CDR.json')).keys())
        keyRcount=0
        for theR in data['labels']:
            if theR['r'] in unseen_reltype:
                keyRcount+=1
        if keyRcount<0:
            print(" *********** No target relation in this Doc 7_1 ***********")
            index += 1
            continue
        data['tag']=doc['relation_tag']
        datasets.append(data)
        index+=1
    dis_generate = {}
    relation_types = json.load(open(relation_prompt))
    for key in relation_types.keys():
        dis_generate[info2rel[key]] = 0
    for item in datasets:
        labels = item['labels']
        for labe in labels:
            if labe['r'] in dis_generate.keys():
                dis_generate[labe['r']] += 1
    print("saving rate: ", (len(datasets) / len(origin)) * 100)
    print("***************** statistic ******************")
    print("len of datasets: ", len(datasets))
    print("avg relational fact: ", (relationfact / len(datasets)) )
    print("avg entity: ", (entityNumber / len(datasets)))
    print("distribution: ", dis_generate)
    datasets=json.dumps(datasets)
    unseenpath=savedor+'CDR_synthetic_data.json'
    with open(unseenpath, 'w') as f:
        f.write(datasets)
    f.close()
    print("save path: ", unseenpath)

def evaluate_data_quality(original_data_path, synthetic_data_path, quality_threshold=3):
    print("Starting synthetic data quality evaluation...")
    with open(original_data_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    with open(synthetic_data_path, 'r', encoding='utf-8') as f:
        synthetic_data = json.load(f)
    reference_samples = random.sample(original_data, min(5, len(original_data)))
    high_quality_data = []
    low_quality_data = []
    for i, synthetic_sample in enumerate(tqdm.tqdm(synthetic_data, desc="Evaluating data quality")):
        try:
            evaluation_prompt = build_evaluation_prompt(reference_samples, synthetic_sample)
            quality_score = get_quality_score(evaluation_prompt)
            if quality_score >= quality_threshold:
                high_quality_data.append(synthetic_sample)
            else:
                low_quality_data.append(synthetic_sample)
            time.sleep(1)
        except Exception as e:
            print(f"Error evaluating sample {i}: {e}")
            high_quality_data.append(synthetic_sample)
    print(
        f"Quality evaluation completed: {len(high_quality_data)} high-quality samples, {len(low_quality_data)} low-quality samples")

    high_quality_path = savedor + 'CDR_high_quality_synthetic_data.json'
    with open(high_quality_path, 'w', encoding='utf-8') as f:
        json.dump(high_quality_data, f, ensure_ascii=False, indent=2)

    low_quality_path = savedor + 'CDR_low_quality_synthetic_data.json'
    with open(low_quality_path, 'w', encoding='utf-8') as f:
        json.dump(low_quality_data, f, ensure_ascii=False, indent=2)

    return high_quality_data


def build_evaluation_prompt(reference_samples, synthetic_sample):
    reference_descriptions = []
    for i, sample in enumerate(reference_samples[:3]):
        desc = f"Reference Sample {i + 1}:\n"
        desc += f"Text: {' '.join([' '.join(sent) for sent in sample['sents']])}\n"
        desc += f"Entity Count: {len(sample['vertexSet'])}\n"
        desc += f"Relation Count: {len(sample['labels'])}\n"
        reference_descriptions.append(desc)

    synthetic_desc = f"Synthetic Sample to Evaluate:\n"
    synthetic_desc += f"Text: {synthetic_sample['sents'] if isinstance(synthetic_sample['sents'], str) else ' '.join([' '.join(sent) for sent in synthetic_sample['sents']])}\n"
    synthetic_desc += f"Entity Count: {len(synthetic_sample['vertexSet'])}\n"
    synthetic_desc += f"Relation Count: {len(synthetic_sample['labels'])}\n"

    prompt = f"""
    You are a biomedical relation extraction data quality evaluation expert. Please evaluate the quality of the synthetic sample based on the following high-quality reference samples.

    {''.join(reference_descriptions)}

    {synthetic_desc}

    Please evaluate quality from the following dimensions (1-10 points, 10 being the highest):
    1. Text fluency and authenticity
    2. Accuracy of entity annotations
    3. Reasonableness of relation extraction
    4. Similarity to real biomedical data
    5. Overall data quality

    Please provide a comprehensive score (integer from 1-10) and briefly explain your reasoning.

    Format:
    Score: [1-10]
    Reasoning: [Brief explanation]
    """

    return prompt

def get_quality_score(prompt):
    messages = [
        {'role': 'system',
         'content': 'You are a professional data quality evaluation expert. Please respond strictly in the required format.'},
        {'role': 'user', 'content': prompt}
    ]

    try:
        response = get_completion_from_messages(messages, temperature=0.1)

        # Parse response to extract score
        lines = response.strip().split('\n')
        for line in lines:
            if line.startswith('Score:'):
                score_str = re.findall(r'\d+', line)
                if score_str:
                    return int(score_str[0])

        return 3

    except Exception as e:
        print(f"Error getting quality score: {e}")
        return 3  # Return medium score on error

def extract_train_triplets(train_data_path):
    try:
        with open(train_data_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        print(f"成功加载训练集，共{len(train_data)}个文档")
    except Exception as e:
        print(f"加载训练集失败：{e}")
        return []

    triplets_dict = {}
    rel_id_to_name = {0: "no chemical disease induction relation", 1: "chemical induced disease"}

    for doc in train_data:
        full_text = reconstruct_text(doc.get("sents", []))
        if not full_text:
            full_text = doc.get("context", "")

        entity_idx_map = {}
        vertex_set = doc.get("vertexSet", [])

        for ent_group_idx, ent_group in enumerate(vertex_set):
            if not ent_group: continue
            first_mention = ent_group[0]
            ent_name_list = first_mention.get("name", [])
            ent_type = first_mention.get("type", "").strip()

            if not isinstance(ent_name_list, list) or len(ent_name_list) == 0: continue
            if ent_type not in ["Chemical", "Disease"]: continue

            ent_key = "_".join(ent_name_list).lower()
            entity_idx_map[ent_group_idx] = (ent_name_list, ent_type, ent_key)

        labels = doc.get("labels", [])
        for label in labels:
            h_idx = label.get("h", -1)
            t_idx = label.get("t", -1)
            rel_id = label.get("r", -1)

            if h_idx not in entity_idx_map or t_idx not in entity_idx_map or rel_id not in rel_id_to_name:
                continue

            h_name_list, h_type, h_key = entity_idx_map[h_idx]
            t_name_list, t_type, t_key = entity_idx_map[t_idx]

            if h_type != "Chemical" or t_type != "Disease": continue

            rel_name = rel_id_to_name[rel_id]
            triplet_key = f"{h_key}||{t_key}||{rel_name}"

            if triplet_key not in triplets_dict:
                triplets_dict[triplet_key] = {
                    "chem": h_name_list,
                    "dis": t_name_list,
                    "rel": rel_name,
                    "original_text": full_text
                }

    train_triplets = list(triplets_dict.values())
    rel_count = {"chemical induced disease": 0, "no chemical disease induction relation": 0}
    for triplet in train_triplets:
        if triplet["rel"] in rel_count:
            rel_count[triplet["rel"]] += 1

    print(f"从训练集提取到{len(train_triplets)}个去重三元组（含原文背景）")
    print(f"三元组关系分布：{rel_count}")

    return train_triplets

def main():
    num=50
    relation_types = json.load(open(relation_prompt))
    train_data_path = "F:/CDR_json格式/convert_train.json"
    train_triplets = extract_train_triplets(train_data_path)
    for i in range(num):
        print("***********",i,"***********")
        generate(relation_types,train_triplets)
    static()
    print("transfering...")
    transfer()

    print("Starting data quality evaluation...")
    original_data_path = "F:/CDR_json格式/convert_train.json"  # Update with your original dataset path
    synthetic_data_path = savedor + 'CDR_synthetic_data.json'
    high_quality_data = evaluate_data_quality(original_data_path, synthetic_data_path)
    print(f"Final high-quality data retained: {len(high_quality_data)} samples")
    print("\n=== Starting random sampling of 50 high-quality samples ===")
    high_quality_path = "F:\GenRDK-main\synthetic_data_m5_s5\CDR_high_quality_synthetic_data.json"
    with open(high_quality_path, 'r', encoding='utf-8') as f:
        high_quality_data = json.load(f)
    sample_count = 500
    if len(high_quality_data) >= sample_count:
        sampled_50_data = random.sample(high_quality_data, sample_count)
        print(f"Successfully sampled {sample_count} samples from {len(high_quality_data)} high-quality samples.")
    else:
        sampled_50_data = high_quality_data
        print(f"High-quality samples ({len(high_quality_data)}) are less than {sample_count}, sampling all.")

    sampled_50_path = savedor + 'CDR_high_quality_500_samples.json'
    with open(sampled_50_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_50_data, f, ensure_ascii=False, indent=2)  # indent=2让JSON格式更易读

    print(f"Sampled 50 (or all) high-quality samples saved to: {sampled_50_path}")
    print(f"Sampled data length: {len(sampled_50_data)}")

main()




