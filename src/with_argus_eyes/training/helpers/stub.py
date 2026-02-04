def stub_name(retriever='', risk_name='', clf_name='', tag=''):
    parts = [p for p in [retriever, risk_name, clf_name, tag] if str(p) != ""]
    return "_".join(map(str, parts))

