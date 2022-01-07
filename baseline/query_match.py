import json

def query_match(gt, pred) : 
    accs = []
    for i in range(len(gt)) : 
      total_len = len(gt[i])
      matched = 0 
      if gt[i][0] == pred[i][0] : matched = matched + 1
      if gt[i][1] == pred[i][1] : matched = matched + 1
      start = 2
      end = start + 3
      gt_string = [] 
      pred_strs = []
      while end < len(gt[i]) :
        cond_str = []
        for k in range(len(gt[i][start:end])) :
            cond_str.append(str(gt[i][start:end][k]))

        gt_string = gt_string + [" ".join(cond_str)]
        start = start + 3
        end = end + 3
      start = 2
      end = start + 3
      pred_str = []
      while end < len(pred[i]) :
        cond_str = []
        for k in range(len(gt[i][start:end])) :
            cond_str.append(str(gt[i][start:end][k]))
        pred_str = pred_str + [" ".join(cond_str)]
        start = start + 3
        end = end + 3
      for k in range(len(pred_str)) :
        if pred_str[k] in gt_string : 
          matched = matched + 3
      accs.append(float(matched * 100 )/float(total_len))

    return float(sum(accs))/float(len(accs))


def evaluate(file1, file2) :
  data_gt = {}
  with open(file1) as f:
      data_gt = json.load(f)
  with open(file2) as f:
      data_pred = json.load(f)

  print(query_match(data_gt['ans'], data_pred['ans']))

evaluate('test_gt.json', 'test_pred.json')
