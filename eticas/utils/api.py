import os
import requests
import pandas as pd
import numpy as np
import json


def get_audit(audit_id=2289):
    """
    Parameters
    ----------
   :param audit_id: The ID of the audit to be explored.
    Returns
    ----------
   :return: A dictionary containing the results for each group
    """
    BASE_URL = os.getenv("ITACA_BASE_URL")
    if not BASE_URL:
        raise ValueError("❌ 'ITACA_BASE_URL' NO DEFINED.")
    API_TOKEN = os.getenv("ITACA_API_TOKEN")
    if not API_TOKEN:
        raise ValueError("❌ 'ITACA_API_TOKEN' NO DEFINED.")

    url = f"{BASE_URL}audit/{audit_id}/"
    headers = {
        "Authorization": f"Token {API_TOKEN}",
        "Accept": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise requests.HTTPError(f"❌ Error {response.status_code}: {response.text}")


def get_departments():
    """
    Returns
    ----------
   :return: A DataFrame with all the departments.
    """
    BASE_URL = os.getenv("ITACA_BASE_URL")
    if not BASE_URL:
        raise ValueError("❌ 'ITACA_BASE_URL' NO DEFINED.")
    API_TOKEN = os.getenv("ITACA_API_TOKEN")
    if not API_TOKEN:
        raise ValueError("❌ 'ITACA_API_TOKEN' NO DEFINED.")
    url = f"{BASE_URL}department/"
    headers = {
        "Authorization": f"Token {API_TOKEN}",
        "Accept": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        raise requests.HTTPError(f"❌ Error {response.status_code}: {response.text}")


def get_models(department=None):
    """
    Parameters
    ----------
   :param department: The ID of the department to be explored.
    Returns
    ----------
   :return: A DataFrame containing all the models for selected department.
    """
    BASE_URL = os.getenv("ITACA_BASE_URL")
    if not BASE_URL:
        raise ValueError("❌ 'ITACA_BASE_URL' NO DEFINED.")
    API_TOKEN = os.getenv("ITACA_API_TOKEN")
    if not API_TOKEN:
        raise ValueError("❌ 'ITACA_API_TOKEN' NO DEFINED.")
    url = f"{BASE_URL}department/{department}/model/"
    headers = {
        "Authorization": f"Token {API_TOKEN}",
        "Accept": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        raise requests.HTTPError(f"❌ Error {response.status_code}: {response.text}")


def get_audits(model=None):
    """
    Parameters
    ----------
   :param model: The ID of the model to be explored.
    Returns
    ----------
   :return: A DataFrame containing the audits.
    """
    BASE_URL = os.getenv("ITACA_BASE_URL")
    if not BASE_URL:
        raise ValueError("❌ 'ITACA_BASE_URL' NO DEFINED.")
    API_TOKEN = os.getenv("ITACA_API_TOKEN")
    if not API_TOKEN:
        raise ValueError("❌ 'ITACA_API_TOKEN' NO DEFINED.")
    url = f"{BASE_URL}audit/?auditable_model_id={model}"
    headers = {
        "Authorization": f"Token {API_TOKEN}",
        "Accept": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        result = response.json()['results']
        return pd.DataFrame(result)
    else:
        raise requests.HTTPError(f"❌ Error {response.status_code}: {response.text}")


def upload_audit(department_id=None,
                 model_id=None,
                 model=None):
    BASE_URL = os.getenv("ITACA_BASE_URL")
    if not BASE_URL:
        raise ValueError("❌ 'ITACA_BASE_URL' NO DEFINED.")
    API_TOKEN = os.getenv("ITACA_API_TOKEN")
    if not API_TOKEN:
        raise ValueError("❌ 'ITACA_API_TOKEN' NO DEFINED.")
    departments = get_departments()
    if departments.shape[0] > 0:
        if department_id not in departments.id.values.tolist():
            raise ValueError("Deparment ID does not exist..")
    else:
        raise ValueError("Deparment ID does not exist..")
    models = get_models(department=department_id)
    if models.shape[0] > 0:
        if model_id not in models.id.values.tolist():
            raise ValueError("Model ID does not exist..")
    else:
        raise ValueError("Model ID does not exist..")
    url = f"{BASE_URL}department/{department_id}/model/{model_id}/audit/result"
    audit_metrics = upload_json_audit(model)
    json_str = json.dumps(audit_metrics)
    files = {
        "metrics": ("archivo.json", json_str, "multipart/form-data")
    }
    headers = {
        "Authorization": f"Token {API_TOKEN}"
    }
    response = requests.post(url, headers=headers, files=files)

    return response.status_code


def overview(key0, key1,
             no_none_share,
             ref_evol):
    result = {}
    for s in no_none_share:
        diff = np.round(s[0]-ref_evol, 2).item()
        result.update({key0+'_'+s[1]+'_difference': abs(diff)})
        result.update({key0+'_'+s[1]+'_evol': 'equal to' if diff == 0 else 'below' if diff < 0 else 'above'})

    if len(no_none_share) > 1:
        for n in range(len(no_none_share)-1):
            diff = no_none_share[n+1][0] - no_none_share[n][0]
            key = f"{key1}_{no_none_share[n][1]}_{no_none_share[n+1][1]}"
            value = ['negative', 'neutral', 'positive'][(diff > 0) - (diff < 0) + 1]
            result.update({key: value})
    return result


def scoring_evolution(first_share,
                      last_share,
                      ref_share):
    if last_share >= ref_share:
        share_risk = 100
    elif last_share < first_share:
        penalty = -20
        norm_share = (last_share - 0) / (100 - 0)
        share_risk = max(np.round(penalty + (norm_share * 100), 4).item(), 0)
    else:
        norm_share = (last_share - 0) / (100 - 0)
        share_risk = max(np.round((norm_share * 100), 4).item(), 0)

    return share_risk


def normalize_benchmarking(X):
    result = X - ((X - 1) * 2) if X > 1 else X
    result = max(result, 0)
    result = result * 100
    return result


def bias_direction(X):
    if X == 1:
        return 'Correct-representation'
    if X < 1:
        return 'Under-representation'
    else:
        return 'Over-representation'


def upload_json_audit(model):
    json_result = model.json_results(norm_values=True)
    audit_result = {}
    for p in json_result.keys():
        if p != 'error':
            if (p.lower() not in ['gender', 'ethnicity', 'age', 'gender_ethnicity']) & ('sensitive_' not in p.lower()):
                p_id = 'sensitive_'+p.lower()
            else:
                p_id = ''+p.lower()
            if p in str(model.distribution_ref):
                ref_aux = model.distribution_ref[p]
            else:
                ref_aux = 50
            audit_result[p_id] = {}
            training_data = json_result.get(p, {}).get('benchmarking',
                                                       {}).get('labeled_da_inconsistency', None)
            training_positive = json_result.get(p, {}).get('benchmarking',
                                                           {}).get('labeled_da_positive', None)
            operational_data = json_result.get(p, {}).get('benchmarking',
                                                          {}).get('operational_da_inconsistency', None)
            operational_positive = json_result.get(p, {}).get('benchmarking',
                                                              {}).get('operational_da_positive', None)
            impact_data = json_result.get(p, {}).get('benchmarking',
                                                     {}).get('impact_da_inconsistency', None)
            impact_positive = json_result.get(p, {}).get('benchmarking',
                                                         {}).get('impact_da_positive', None)

            audit_result[p_id]['benchmarking'] = {
                'ref': ref_aux,
                'training_data': training_data,
                'training_positive': training_positive,
                'operational_data': operational_data,
                'operational_positive':  operational_positive,
                'impact_data': impact_data,
                'impact_positive':  impact_positive,

                'training_data_risk': normalize_benchmarking(
                    0 if training_data is None else training_data/ref_aux),
                'training_positive_risk':  normalize_benchmarking(
                    0 if training_positive is None else training_positive/ref_aux),
                'operational_data_risk': normalize_benchmarking(
                    0 if operational_data is None else operational_data/ref_aux),
                'operational_positive_risk':  normalize_benchmarking(
                    0 if operational_positive is None else operational_positive/ref_aux),
                'impact_data_risk': normalize_benchmarking(
                    0 if impact_data is None else impact_data/ref_aux),
                'impact_positive_risk': normalize_benchmarking(
                    0 if impact_positive is None else impact_positive/ref_aux),

                'training_data_direction': bias_direction(
                    0 if training_data is None else training_data/ref_aux),
                'training_positive_direction':  bias_direction(
                    0 if training_positive is None else training_positive/ref_aux),
                'operational_data_direction': bias_direction(
                    0 if operational_data is None else operational_data/ref_aux),
                'operational_positive_direction':  bias_direction(
                    0 if operational_positive is None else operational_positive/ref_aux),
                'impact_data_direction': bias_direction(
                    0 if impact_data is None else impact_data/ref_aux),
                'impact_positive_direction': bias_direction(
                    0 if impact_positive is None else impact_positive/ref_aux),
                }

            audit_result[p_id]['distribution'] = {
                'ref': 80,
                'training_proxy': json_result.get(p,
                                                  {}).get('distribution',
                                                          {}).get('labeled_dxa_inconsistency', None),
                'training_label': json_result.get(p,
                                                  {}).get('distribution',
                                                          {}).get('labeled_da_informative', None),
                'operational_proxy': json_result.get(p,
                                                     {}).get('distribution',
                                                             {}).get('operational_dxa_inconsistency', None),
                'operational_label': json_result.get(p,
                                                     {}).get('distribution',
                                                             {}).get('operational_da_informative', None),
                'impact_proxy': json_result.get(p,
                                                {}).get('distribution',
                                                        {}).get('impact_dxa_inconsistency', None),
                'impact_label': json_result.get(p,
                                                {}).get('distribution',
                                                        {}).get('impact_da_informative', None),
                }
            audit_result[p_id]['fairness'] = {
                'ref': 80,

                'training_DI': json_result.get(p,
                                               {}).get('fairness',
                                                       {}).get('labeled_DI', None),
                'operational_DI': json_result.get(p,
                                                  {}).get('fairness',
                                                          {}).get('operational_DI', None),
                'impact_DI': json_result.get(p,
                                             {}).get('fairness',
                                                     {}).get('impact_DI', None),

                'training_SPD': json_result.get(p,
                                                {}).get('fairness',
                                                        {}).get('labeled_SPD', None),
                'operational_SPD': json_result.get(p,
                                                   {}).get('fairness',
                                                           {}).get('operational_SPD', None),
                'impact_SPD': json_result.get(p,
                                              {}).get('fairness',
                                                      {}).get('impact_SPD', None),

                'training_TPR': json_result.get(p,
                                                {}).get('fairness',
                                                        {}).get('labeled_TPR', None),
                'training_FPR': json_result.get(p,
                                                {}).get('fairness',
                                                        {}).get('labeled_FPR', None),
                'training_PPV': json_result.get(p,
                                                {}).get('fairness',
                                                        {}).get('labeled_PPV', None),
                'training_PNV': json_result.get(p,
                                                {}).get('fairness',
                                                        {}).get('labeled_PNV', None),

                'training_equality': json_result.get(p,
                                                     {}).get('fairness',
                                                             {}).get('labeled_equality', None),
                'training_equity': json_result.get(p,
                                                   {}).get('fairness',
                                                           {}).get('labeled_equity', None),
                'operational_equality': json_result.get(p,
                                                        {}).get('fairness',
                                                                {}).get('operational_equality', None),
                'operational_equity': json_result.get(p,
                                                      {}).get('fairness',
                                                              {}).get('operational_equity', None),
                'impact_equality': json_result.get(p,
                                                   {}).get('fairness',
                                                           {}).get('impact_equality', None),
                'impact_equity': json_result.get(p,
                                                 {}).get('fairness',
                                                         {}).get('impact_equity', None),

                }
            audit_result[p_id]['impact_ratio'] = {}
            audit_result[p_id]['impact_ratio']['training'] = {
                'ref': 80,
                'df_result': model.labeled_results.get('impact_ratio',
                                                       {}).get(p, {}).get('df_result', None),
                'pass': model.labeled_results.get('impact_ratio',
                                                  {}).get(p, {}).get('pass', None),
                'total': model.labeled_results.get('impact_ratio',
                                                   {}).get(p, {}).get('total', None),
                'total_filter': model.labeled_results.get('impact_ratio',
                                                          {}).get(p, {}).get('total_filter', None),
                'unknown': model.labeled_results.get('impact_ratio',
                                                     {}).get(p, {}).get('unknown', None),
                }
            audit_result[p_id]['impact_ratio']['operational'] = {
                'ref': 80,
                'df_result': model.production_results.get('impact_ratio',
                                                          {}).get(p, {}).get('df_result', None),
                'pass': model.production_results.get('impact_ratio',
                                                     {}).get(p, {}).get('pass', None),
                'total': model.production_results.get('impact_ratio',
                                                      {}).get(p, {}).get('total', None),
                'total_filter': model.production_results.get('impact_ratio',
                                                             {}).get(p, {}).get('total_filter', None),
                'unknown': model.production_results.get('impact_ratio',
                                                        {}).get(p, {}).get('unknown', None),
                }
            audit_result[p_id]['impact_ratio']['impacted'] = {
                'ref': 80,
                'df_result': model.impacted_results.get('impact_ratio',
                                                        {}).get(p, {}).get('df_result', None),
                'pass': model.impacted_results.get('impact_ratio',
                                                   {}).get(p, {}).get('pass', None),
                'total': model.impacted_results.get('impact_ratio',
                                                    {}).get(p, {}).get('total', None),
                'total_filter': model.impacted_results.get('impact_ratio',
                                                           {}).get(p, {}).get('total_filter', None),
                'unknown': model.impacted_results.get('impact_ratio',
                                                      {}).get(p, {}).get('unknown', None),
                }

            audit_result[p_id]['performance'] = {
                'ref': 80,
                'inconsistency': json_result.get(p,
                                                 {}).get('drift', {}).get('drift', None),
                'poor_performance': json_result.get(p,
                                                    {}).get('performance', {}).get('poor_performance', None),
                'recall': json_result.get(p,
                                          {}).get('performance', {}).get('recall', None),
                'f1_score': json_result.get(p,
                                            {}).get('performance',  {}).get('f1_score', None),
                'accuracy': json_result.get(p,
                                            {}).get('performance', {}).get('accuracy', None),
                'precision': json_result.get(p,
                                             {}).get('performance', {}).get('precision', None),
                'TP': json_result.get(p,
                                      {}).get('performance', {}).get('TP', None),
                'FP': json_result.get(p,
                                      {}).get('performance', {}).get('FP', None),
                'TN': json_result.get(p,
                                      {}).get('performance', {}).get('TN', None),
                'FN': json_result.get(p,
                                      {}).get('performance', {}).get('FN', None),

                }
            files = ['training', 'operational', 'impact']

            evolution_params = {'score_first_last': {
                'key': 'benchmarking',
                'fields': ['training_data', 'operational_data', 'impact_data'],
                'overview': {'key0': 'share', 'key1': 'evol'},
                },
                'score_positives_first_last': {
                'key': 'benchmarking',
                'fields': ['training_positive', 'operational_positive', 'impact_positive'],
                'overview': {'key0': 'share_positives', 'key1': 'evol_positives'},
                },
                'score_DI_first_last': {
                'key': 'fairness',
                'fields': ['training_DI', 'operational_DI', 'impact_DI'],
                'overview': {'key0': 'share_DI', 'key1': 'evol_DI'},
                },
                'score_EQA_first_last': {
                'key': 'fairness',
                'fields': ['training_equality', 'operational_equality', 'impact_equality'],
                'overview': {'key0': 'share_EQA', 'key1': 'evol_EQA'},
                },
                'score_EQI_first_last': {
                'key': 'fairness',
                'fields': ['training_equity', 'operational_equity', 'impact_equity'],
                'overview': {'key0': 'share_EQI', 'key1': 'evol_EQI'},
                },
                'score_proxy_first_last': {
                'key': 'distribution',
                'fields': ['training_proxy', 'operational_proxy', 'impact_proxy'],
                'overview': {'key0': 'share_proxy', 'key1': 'evol_proxy'},
                },
                'score_label_first_last': {
                'key': 'distribution',
                'fields': ['training_label', 'operational_label', 'impact_label'],
                'overview': {'key0': 'share_label', 'key1': 'evol_label'},
                }}
            audit_result[p_id]['scoring_evolution'] = {}
            audit_result[p_id]['overview'] = {}
            for evolution in evolution_params.items():
                share_values = [audit_result[p_id][evolution[1]['key']][f] for f in evolution[1]['fields']]

                ref_evol = audit_result[p_id][evolution[1]['key']]['ref']
                no_none_share = [(v, f) for f, v in zip(files, share_values) if v is not None]
                if len(no_none_share) == 0:
                    share_first_last = [0] + [ref_evol]
                elif len(no_none_share) == 1:
                    share_first_last = [no_none_share[0][0]] + [ref_evol]
                else:
                    share_first_last = [no_none_share[0][0]] + [no_none_share[-1][0]]

                audit_result[p_id]['scoring_evolution'][evolution[0]] = scoring_evolution(share_first_last[0],
                                                                                          share_first_last[1],
                                                                                          ref_evol)
                audit_result[p_id]['overview'].update(overview(evolution[1]['overview']['key0'],
                                                               evolution[1]['overview']['key1'],
                                                               no_none_share,
                                                               ref_evol))
    result = {
        'detail': {
            'training_detail': model.labeled_results,
            'operational_detail': model.production_results,
            'impact_detail': model.impacted_results,
            'drift_detail': model.drift_results
        },
        'audit_result': audit_result
    }

    return result
