

def dict_list_to_list_dict(data):
  if not isinstance(data, dict) or not all(isinstance(value, list) for value in data.values()):
    raise ValueError("Input must be a dictionary with lists as values")

  keys = list(data.keys())
  result = []
  for i in range(len(data[keys[0]])):
    new_dict = {}
    for key, value_list in data.items():
      new_dict[key] = value_list[i]
    result.append(new_dict)

  return result


def trim_string(text, max_words=100):
  words = text.split()
  if len(words) <= max_words:
    return text

  trimmed_words = words[:max_words]
  return " ".join(trimmed_words)