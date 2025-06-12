from automatic_prompt_engineer import llm, data, evaluate
import numpy as np
from tqdm import tqdm

special_output_token = '[[[[OUTPUT]]]]'


def get_query(prompt, eval_template, input_, output_, demo_data, demos_template):
    """
    Returns the text sent to the LLM for likelihood evaluation.
    Parameters:
        prompt: The prompt.
        eval_template: The template for the evaluation queries.
        input_: The input.
        output_: The output.
    Returns:
        The query for the LLM and the range of the output text in the form of (start_idx, end_idx).
    """
    demos = demos_template.fill(demo_data)
    query = eval_template.fill(prompt=prompt,
                               input=input_,
                               output=output_,
                               full_demo=demos)
    query_without_output = eval_template.fill(prompt=prompt,
                                              input=input_,
                                              output=special_output_token,
                                              full_demo=demos)

    first_idx = query_without_output.find(special_output_token)
    output_idx = first_idx, first_idx + len(output_)
    return query, output_idx


def accuracy_evaluator(prompts, eval_template, eval_data, demos_template, few_shot_data, config, client):
    """
    Dummy accuracy check
    """
    queries = []
    targets = []
    prompt_list = []
    for prompt in prompts:
        subsampled_data = data.subsample_data(
            eval_data, config['num_samples'])
        for d in zip(*subsampled_data):
            input_, output_ = d
            demo_data = data.subsample_data(
                few_shot_data, config['num_few_shot'])
            query, output_idx = get_query(
                prompt, eval_template, input_, output_, demo_data, demos_template)
            queries.append(query[:output_idx[0]])
            target = query[output_idx[0]:output_idx[1]].split(',')
            target = [t.strip().lower() for t in target] 
            targets.append(target)
            prompt_list.append(prompt)

    model = llm.model_from_config(config['model'], client)
    accs = []
    # TODO: parallelize? 
    prev_prompt = None
    for prompt, query, target in tqdm(zip(prompt_list, queries, targets)): 
        if prev_prompt is None or prev_prompt != prompt:
            accs.append([])
        answer = model.generate_text(query, n=1)[0].lower()
        # print('model said:', answer)
        # print('target:', target)
        acc = 0
        for t in target: 
            acc += t in answer
        acc /= len(target)
        # print('acc:', acc, 'prompt:', prompt)
        accs[-1].append(acc)
        prev_prompt = prompt

    # print(accs)
    # for prompt, acc in zip(prompts, accs): 
    #     print(prompt, acc)

    res = AccuracyEvaluationResult(prompts, accs, config['num_samples'])

    return res


class AccuracyEvaluationResult(evaluate.EvaluationResult):
    """
    A class for storing the results of an accuracy evaluation. Supports
    sorting prompts by various statistics.
    """

    def __init__(self, prompts, accs, num_samples):
        self.prompts = prompts
        self.prompt_accs = accs
        self.num_samples = num_samples

    def _agg_accs(self, method):
        """For each prompt, compute a statistic of the likelihoods (e.g., mean, median, etc.)"""
        if method == 'mean':
            return [np.mean(lps) for lps in self.prompt_accs]
        elif method == 'median':
            return [np.median(lps) for lps in self.prompt_accs]
        elif method == 'std':
            return [np.std(lps) for lps in self.prompt_accs]
        elif method == 'max':
            return [np.max(lps) for lps in self.prompt_accs]
        elif method == 'min':
            return [np.min(lps) for lps in self.prompt_accs]
        elif method == 'iqm':
            return [np.mean(np.percentile(lps, [25, 75])) for lps in self.prompt_accs]
        else:
            raise ValueError(
                f'Unknown method {method} for aggregating likelihoods')

    def sorted(self, method='default'):
        if method == 'default':
            scores = self._agg_accs('mean')
        else:
            scores = self._agg_accs(method)
        # Sort prompts by score
        sorted_prompts = [p for _, p in sorted(zip(scores, self.prompts))]
        sorted_scores = sorted(scores)
        # Reverse both and convert to lists
        sorted_prompts = list(reversed(sorted_prompts))
        sorted_scores = list(reversed(sorted_scores))
        return sorted_prompts, sorted_scores

    def in_place(self, method='default'):
        if method == 'default':
            scores = self._agg_accs('mean')
        else:
            scores = self._agg_accs(method)
        return self.prompts, scores

    def __str__(self):
        s = ''
        prompts, scores = self.sorted()
        s += 'acc: prompt\n'
        s += '----------------\n'
        for prompt, score in list(zip(prompts, scores))[:10]:
            s += f'{score:.2f}: {prompt}\n'
        return s
