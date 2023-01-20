from github.downstream_tasks.PepBCL import protdefault
def binding_predict(config, chooseid):
    table_data = {}
    table_data["type"] = 'prot'
    input, pred_class, binding_index, pdbid, chain = protdefault.main(config, chooseid)
    table_data["prot_seq"] = input
    table_data["lable"] = pred_class
    table_data["binding_index"] = binding_index
    table_data["pdbid"] = pdbid
    table_data["chain"] = chain

    table_data = str(table_data)
    table_data = eval(table_data)

    return table_data