import json
from collections import Counter


# check if there is an exact match
def exact_match(pred_sentence, gold_sentence, src_sentence):

  if pred_sentence.lower() == gold_sentence.lower():
    return 1
  else:
    return 0


# check if first NP matches
def first_word(pred_sentence, gold_sentence, src_sentence):
  pred_words = pred_sentence.split()
  gold_words = gold_sentence.split()
  if len(pred_words) > 0 and pred_words[0].lower() == gold_words[0].lower():
    return 1
  return 0

def second_word(pred_sentence, gold_sentence, src_sentence):
  pred_words = pred_sentence.split()
  gold_words = gold_sentence.split()
  if len(pred_words) > 1 and pred_words[1].lower() == gold_words[1].lower():
    return 1
  return 0


QUESTION_AUXILIARIES = set(["have", "haven't", "has", "hasn't", "hat", "haben", "ist", "sind", "kann", "k\xf6nnen"])

def three_auxiliaries(pred_sentence, gold_sentence, src_sentence):
    pred_words = pred_sentence.split()
    aux_count = 0
    for word in pred_words:
        if word in QUESTION_AUXILIARIES:
            aux_count += 1

    if aux_count > 2:
        return 1

    return 0


# different auxiliary metrics

def delete_first_prepose_first(pred_sentence, gold_sentence, src_sentence):
    pred_sentence.replace(",", " ,").replace("?", " ?").replace(".", " .").replace("  ", " ")
    pred_words = pred_sentence.split()

    if len(pred_words) < 0 or pred_words[0] not in QUESTION_AUXILIARIES:
      return 0


    pred_aux = []
    for word in pred_words:
        if word in QUESTION_AUXILIARIES:
          pred_aux.append(word)

    if len(pred_aux) != 2:
      return 0

    src_words = src_sentence.split()
    src_aux = []
    for word in src_words:
        if word in QUESTION_AUXILIARIES:
          src_aux.append(word)

    if pred_aux == src_aux:
      return 1

    return 0

def prepose_first(pred_sentence, gold_sentence, src_sentence):
    pred_sentence.replace(",", " ,").replace("?", " ?").replace(".", " .").replace("  ", " ")
    pred_words = pred_sentence.split()

    if len(pred_words) < 0 or pred_words[0] not in QUESTION_AUXILIARIES:
      return 0


    pred_aux = []
    for word in pred_words:
        if word in QUESTION_AUXILIARIES:
          pred_aux.append(word)


    src_words = src_sentence.split()
    src_aux = []
    for word in src_words:
        if word in QUESTION_AUXILIARIES:
          src_aux.append(word)

    if pred_aux[0] == src_aux[0]:
      return 1

    return 0



def delete_main_prepose_main(pred_sentence, gold_sentence, src_sentence):
    pred_sentence.replace(",", " ,").replace("?", " ?").replace(".", " .").replace("  ", " ")
    pred_words = pred_sentence.split()

    if len(pred_words) < 0 or pred_words[0] not in QUESTION_AUXILIARIES:
      return 0

    pred_aux = []
    for word in pred_words:
        if word in QUESTION_AUXILIARIES:
          pred_aux.append(word)

    if len(pred_aux) != 2:
      return 0

    gold_words = gold_sentence.split()
    gold_aux = []
    for word in gold_words:
        if word in QUESTION_AUXILIARIES:
          gold_aux.append(word)

    if pred_aux == gold_aux:
      return 1

    return 0


def delete_none_prepose_first(pred_sentence, gold_sentence, src_sentence):
    pred_sentence.replace(",", " ,").replace("?", " ?").replace(".", " .").replace("  ", " ")
    pred_words = pred_sentence.split()

    if len(pred_words) < 0 or pred_words[0] not in QUESTION_AUXILIARIES:
      return 0

    pred_aux = []
    for word in pred_words:
        if word in QUESTION_AUXILIARIES:
          pred_aux.append(word)

    if len(pred_aux) != 3:
      return 0

    src_words = src_sentence.split()
    src_aux = []
    for word in src_words:
        if word in QUESTION_AUXILIARIES:
          src_aux.append(word)

    src_aux = [src_aux[0]] + src_aux

    if pred_aux == src_aux:
      return 1

    return 0

def delete_none_prepose_main(pred_sentence, gold_sentence, src_sentence):
    pred_sentence.replace(",", " ,").replace("?", " ?").replace(".", " .").replace("  ", " ")
    pred_words = pred_sentence.split()

    if len(pred_words) < 0 or pred_words[0] not in QUESTION_AUXILIARIES:
      return 0

    pred_aux = []
    for word in pred_words:
        if word in QUESTION_AUXILIARIES:
          pred_aux.append(word)

    if len(pred_aux) != 3:
      return 0

    gold_words = gold_sentence.split()
    gold_aux = []
    for word in gold_words:
        if word in QUESTION_AUXILIARIES:
          gold_aux.append(word)

    gold_aux.append(gold_aux[0])
    if pred_aux == gold_aux:
      return 1

    return 0


PASSIVE_AUXILIARIES = set(["was", "were", "wurde", "wurden"])

# check if NP before passive verb matches
def passive_first_np(pred_sentence, gold_sentence, src_sentence):
  # remove comma
  pred_sentence = pred_sentence.replace(",", "").replace("  ", " ")
  gold_sentence = gold_sentence.replace(",", "").replace("  ", " ")
  pred_words = pred_sentence.split()
  gold_words = gold_sentence.split()
  idx = -1
  for i, word in enumerate(gold_words):
    if word in PASSIVE_AUXILIARIES:
      idx = i

  if idx > 0:
    pred_first_np = " ".join(pred_words[0:idx]).lower()
    gold_first_np = " ".join(gold_words[0:idx]).lower()
    if pred_first_np == gold_first_np:
      return 1

  return 0

PASSIVE_PREPOSITIONS = set(["by", "von"])

DETERMINERS = set(["die", "eine", "meine", "deine", "unsere", "ihre", "einige", "dem", "einem", "meinem", "deinem", "unserem", "ihrem", "der", "einer", "meiner", "deiner", "unserer", "ihrer", "einigen", "den", "meinen", "deinen", "unseren", "ihren", "the", "some", "her", "my", "your", "our"])

# check if NP after passive verb matches
def passive_second_np(pred_sentence, gold_sentence, src_sentence):

  pred_sentence = pred_sentence.replace(",", "").replace("  ", " ")
  gold_sentence = gold_sentence.replace(",", "").replace("  ", " ")
  pred_words = pred_sentence.split()
  gold_words = gold_sentence.split()
  is_german = "von" in gold_words


  aux_idx_gold = -1
  aux_idx_pred = -1
  idx_gold = -1
  idx_pred = -1
  for i, word in enumerate(gold_words):
    if word in PASSIVE_AUXILIARIES:
      aux_idx_gold = i

  for i, word in enumerate(pred_words):
    if word in PASSIVE_AUXILIARIES:
      aux_idx_pred = i

  if aux_idx_gold > 0 and aux_idx_pred > 0:
    if gold_words[aux_idx_gold + 1] in PASSIVE_PREPOSITIONS:
      idx_gold = aux_idx_gold + 2
    elif gold_words[aux_idx_gold + 2] in PASSIVE_PREPOSITIONS:
      idx_gold = aux_idx_gold + 3


    for i, word in enumerate(pred_words[aux_idx_pred+1:]):
      idx = i + aux_idx_pred + 1
      if word in DETERMINERS:
        idx_pred = idx
        break


    if idx_pred > 0 and idx_gold > 0:
      np_len = len(gold_words) - 1 - idx_gold
      # German gold sentence will have the verb after the second NP
      if is_german:
        np_len = np_len - 1

      pred_second_np = " ".join(pred_words[idx_pred:idx_pred+np_len]).lower()
     # print(pred_second_np)
      gold_second_np = " ".join(gold_words[idx_gold:idx_gold+np_len]).lower()
     # print(gold_second_np)
     # print("------------------")
      if gold_second_np == pred_second_np:
        return 1
  return 0

def passive_second_np_no_pp(pred_sentence, gold_sentence, src_sentence):

  pred_sentence = pred_sentence.replace(",", "").replace("  ", " ")
  gold_sentence = gold_sentence.replace(",", "").replace("  ", " ")
  pred_words = pred_sentence.split()
  gold_words = gold_sentence.split()
  is_german = "von" in gold_words


  aux_idx_gold = -1
  aux_idx_pred = -1
  idx_gold = -1
  idx_pred = -1
  for i, word in enumerate(gold_words):
    if word in PASSIVE_AUXILIARIES:
      aux_idx_gold = i

  for i, word in enumerate(pred_words):
    if word in PASSIVE_AUXILIARIES:
      aux_idx_pred = i

  if aux_idx_gold > 0 and aux_idx_pred > 0:
    if gold_words[aux_idx_gold + 1] in PASSIVE_PREPOSITIONS:
      idx_gold = aux_idx_gold + 2
    elif gold_words[aux_idx_gold + 2] in PASSIVE_PREPOSITIONS:
      idx_gold = aux_idx_gold + 3


    for i, word in enumerate(pred_words[aux_idx_pred+1:]):
      idx = i + aux_idx_pred + 1
      if word in DETERMINERS:
        idx_pred = idx
        break


    if idx_pred and idx_gold > 0:

      pred_second_np = " ".join(pred_words[idx_pred:idx_pred+2]).lower()
     # print(pred_second_np)
      gold_second_np = " ".join(gold_words[idx_gold:idx_gold+2]).lower()
     # print(gold_second_np)
     # print("------------------")
      if gold_second_np == pred_second_np:
        return 1
  return 0


def move_second_noun(pred_sentence, gold_sentence, src_sentence):
  pred_sentence = pred_sentence.replace(",", "").replace("  ", " ")
  src_sentence = src_sentence.replace(",", "").replace("  ", " ")
  pred_words = pred_sentence.split()
  src_words = src_sentence.split()

  second_noun = None
  for i, word in enumerate(src_words):
    if i == 0:
      continue
    if word in DETERMINERS:
      second_noun = src_words[i+1]
      break

  if second_noun is None:
    return 0


  if pred_words[1] == second_noun:
    return 1

  return 0

def passive_aux_present(pred_sentence, gold_sentence, src_sentence):
  pred_words = pred_sentence.split()
  for i, word in enumerate(pred_words):
    if word in PASSIVE_AUXILIARIES:
      return 1

  return 0


DETERMINER_EQUIVALENCIES = {
  # Singular
  "der": ["der", "dem", "den"],
  "dem": ["der", "dem", "den"],
  "den": ["die", "der", "dem", "den"], #includes plural
  "ein": ["ein", "einem", "einen"],
  "einem": ["ein", "einem", "einen"],
  "einen": ["ein", "einem", "einen"],
  "mein": ["mein", "meinem", "meinen"],
  "meinem": ["mein", "meinem", "meinen"],
  "meinen": ["mein", "meine", "meinem", "meinen"], #includes plural
  "dein": ["dein", "deinem", "deinen"],
  "deinem": ["dein", "deinem", "deinen"],
  "deinen": ["dein", "deine", "deinem", "deinen"], #includes plural
  "unser": ["unser", "unserem", "unseren"],
  "unserem": ["unser", "unserem", "unseren"],
  "unseren": ["unser", "unsere", "unserem", "unseren"], #includes plural
  "ihr": ["ihr", "ihrem", "ihren"],
  "ihrem": ["ihr", "ihrem", "ihren"],
  "ihren": ["ihr", "ihre", "ihrem", "ihren"], #includes plural
  # Plural
   "die": ["die", "den"],
   "einige": ["einige", "einigen"],
   "einigen": ["einige", "einigen"],
   "meine": ["meine", "meinen"],
   "deine": ["deine", "deinen"],
   "unsere": ["unsere", "unseren"],
   "ihre": ["ihre", "ihren"]
}

NOUN_EQUIVALENCES = {
 "Molch": ["Molch"],
 "Löwe": ["Löwe", "Löwen"],
 "Löwen": ["Löwe", "Löwen"],
 "Pfau": ["Pfau"],
 "Kater": ["Kater"],
 "Rabe": ["Rabe", "Raben"],
 "Raben": ["Rabe", "Raben"],
 "Salamander": ["Salamander", "Salamandern"], #includes plural
 "Dinosaurier": ["Dinosaurier", "Dinosauriern"], #includes plural
 "Papagei": ["Papagei"],
 "Geier": ["Geier", "Geiern"], #includes  plural
 "Wellensittich": ["Wellensittich"],
 "Esel": ["Esel", "Eseln"], #includes  plural
 "Hund": ["Hund"],
 "Ziesel": ["Ziesel", "Zieseln"], #includes  plural

 # plural
  "Molche": ["Molche", "Molchen"],
  "Molchen": ["Molche", "Molchen"],
  "Löwen": ["Löwen"],
  "Pfaue": ["Pfaue", "Pfauen"],
  "Pfauen": ["Pfaue", "Pfauen"],
  "Kater": ["Kater", "Katern"],
  "Katern": ["Kater", "Katern"],
  "Raben": ["Raben"],
  "Salamandern": ["Salamander", "Salamandern"],
  "Dinosauriern": ["Dinosaurier", "Dinosauriern"],
  "Papageie": ["Papageie", "Papageien"],
  "Papageien": ["Papageie", "Papageien"],
  "Geiern": ["Geier", "Geiern"],
  "Wellensittiche": ["Wellensittiche", "Wellensittichen"],
  "Wellensittichen": ["Wellensittiche", "Wellensittichen"],
  "Eseln": ["Esel", "Eseln"],
  "Hunde": ["Hunde", "Hunden"],
  "Hunden": ["Hunde", "Hunden"],
  "Zieseln": ["Ziesel", "Zieseln"]
}



# returns 1 if the correct noun has been moved to subject position
# but the case marking on determiner or noun is incorrect
def first_np_case_incorrect(pred_sentence, gold_sentence, src_sentence):
  pred_sentence = pred_sentence.replace(",", "").replace("  ", " ")
  gold_sentence = gold_sentence.replace(",", "").replace("  ", " ")
  pred_words = pred_sentence.split()
  gold_words = gold_sentence.split()

  if len(pred_words) > 1 and pred_words[0] in DETERMINER_EQUIVALENCIES and pred_words[1] in NOUN_EQUIVALENCES:
      if pred_words[0] != gold_words[0] or pred_words[1] != gold_words[1]:
          if pred_words[0] in DETERMINER_EQUIVALENCIES[gold_words[0]] and pred_words[1] in NOUN_EQUIVALENCES[gold_words[1]]:
              return 1

  return 0

# returns 1 if the correct noun has been moved to the PP
# but the case marking on the determiner and/or noun is incorrect
def second_np_case_incorrect(pred_sentence, gold_sentence, src_sentence):
  pred_sentence = pred_sentence.replace(",", "").replace("  ", " ")
  gold_sentence = gold_sentence.replace(",", "").replace("  ", " ")
  pred_words = pred_sentence.split()
  gold_words = gold_sentence.split()


  aux_idx_gold = -1
  aux_idx_pred = -1
  idx_gold = -1
  idx_pred = -1
  for i, word in enumerate(gold_words):
    if word in PASSIVE_AUXILIARIES:
      aux_idx_gold = i

  for i, word in enumerate(pred_words):
    if word in PASSIVE_AUXILIARIES:
      aux_idx_pred = i

  if aux_idx_gold > 0 and aux_idx_pred > 0:
    if gold_words[aux_idx_gold + 1] in PASSIVE_PREPOSITIONS:
      idx_gold = aux_idx_gold + 2
    elif gold_words[aux_idx_gold + 2] in PASSIVE_PREPOSITIONS:
      idx_gold = aux_idx_gold + 3


    for i, word in enumerate(pred_words[aux_idx_pred+1:]):
      idx = i + aux_idx_pred + 1
      if word in DETERMINERS:
        idx_pred = idx
        break


    if idx_pred > 0 and idx_gold > 0 and idx_pred < len(pred_words) - 1:
      gold_det = gold_words[idx_gold]
      pred_det = pred_words[idx_pred]
      gold_noun = gold_words[idx_gold + 1]
      pred_noun = pred_words[idx_pred + 1]
      if pred_det in DETERMINER_EQUIVALENCIES and pred_noun in NOUN_EQUIVALENCES:
          if gold_det != pred_det or gold_noun != pred_noun:
              if pred_det in DETERMINER_EQUIVALENCIES[gold_det] and pred_noun in NOUN_EQUIVALENCES[gold_noun]:
                return 1
  return 0

def first_np_ignore_case(pred_sentence, gold_sentence, src_sentence):
  pred_sentence = pred_sentence.replace(",", "").replace("  ", " ")
  gold_sentence = gold_sentence.replace(",", "").replace("  ", " ")
  pred_words = pred_sentence.split()
  gold_words = gold_sentence.split()
  idx = 2

  if idx > 0:
    pred_first_np = " ".join(pred_words[0:idx]).lower()
    gold_first_np = " ".join(gold_words[0:idx]).lower()
    if pred_first_np == gold_first_np:
      return 1

  # if it is not an exact match, check if it is a match if one ignores case
  return first_np_case_incorrect(pred_sentence, gold_sentence, src_sentence)


VERB_PARTICIPLE_MAPPING = {
  "unterhielt": "unterhalten",
  "amüsierte": "amüsiert",
  "nervte": "genervt",
  "erfreute": "erfreut",
  "verwirrte": "verwirrt",
  "bewunderte": "bewundert",
  "akzeptierte": "akzeptiert",
  "bedauerte": "bedauert",
  "tröstete": "getröstet",
  "unterhielten": "unterhalten",
  "amüsierten": "amüsiert",
  "nervten": "genervt",
  "erfreuten": "erfreut",
  "verwirrten": "verwirrt",
  "bewunderten": "bewundert",
  "akzeptierten": "akzeptiert",
  "bedauerten": "bedauert",
  "trösteten": "getröstet"
}

def tense_reinflection(pred_sentence, gold_sentence, src_sentence):
  pred_sentence = pred_sentence.replace(",", "").replace("  ", " ")
  src_sentence = src_sentence.replace(",", "").replace("  ", " ")
  pred_words = pred_sentence.split()
  src_words = src_sentence.split()
  for word in src_words:
      if word in VERB_PARTICIPLE_MAPPING:
          if VERB_PARTICIPLE_MAPPING[word] in pred_words:
              return 1
          break

  return 0



def identity(pred_sentence, gold_sentence, src_sentence):
    pred_sentence = pred_sentence.replace(",", "").replace("  ", " ")
    src_sentence = src_sentence.replace(",", "").replace("  ", " ")
    if pred_sentence.lower() == src_sentence.lower():
        print(pred_sentence)
        print(src_sentence)
        print("IDENT")
        print("---------")
        return 1
    return 0


METRIC_FUNCTIONS = {
  "exact_match": exact_match,
  "first_word": first_word,
  "second_word": second_word,
  "prepose_first": prepose_first,
  "three_aux": three_auxiliaries,
  "first_np": passive_first_np,
  "second_np": passive_second_np,
  "second_np_no_pp": passive_second_np_no_pp,
  "passive_aux_present": passive_aux_present,
  "move_second_noun": move_second_noun,
  "identity": identity,
  "delete_first_prepose_first": delete_first_prepose_first,
  "delete_none_prepose_first": delete_none_prepose_first,
  "delete_main_prepose_main": delete_main_prepose_main,
  "delete_none_prepose_main": delete_none_prepose_main,
  "first_np_ignore_case": first_np_ignore_case,
  "first_np_case_incorrect": first_np_case_incorrect,
  "second_np_case_incorrect": second_np_case_incorrect,
  "tense_reinflection": tense_reinflection
  }

def compute_metrics(metrics, pred_file, gold_file, prefix=None):
  with open(pred_file, "r") as pred_f, open(gold_file) as gold_f:
    pred_lines = pred_f.readlines()
    gold_lines = gold_f.readlines()

    total = 0.0
    correct = Counter()
    for i in range(len(pred_lines)):
      pred_line = pred_lines[i].strip()
      if gold_file.endswith(".json"):
        gold_json = json.loads(gold_lines[i])
        if prefix is not None and gold_json["translation"]["prefix"] != prefix:
            continue
        gold_line = gold_json["translation"]["tgt"]
        src_line = gold_json["translation"]["src"]
      else:
        gold_line = gold_lines[i].strip().split("\t")[1]
      # add space before period/question mark/comma
      pred_line = pred_line.replace("?", " ?").replace(".", " .").replace(",", " ,").replace("  ", " ")

      total +=1

      for metric in metrics:
        correct[metric] += METRIC_FUNCTIONS[metric](pred_line, gold_line, src_line)

    for metric in metrics:
      correct[metric] = correct[metric] / total

  return correct
