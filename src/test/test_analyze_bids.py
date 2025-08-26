import pandas as pd
from test_analyze_candidates import load_form_submissions

def test_load_forms_returns_dataframe():

    result = load_form_submissions()
    
    assert isinstance(result, pd.DataFrame)
    assert not result.empty