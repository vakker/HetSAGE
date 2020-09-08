import argparse
import json
from os import path as osp

import ipdb
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import yaml
from tqdm import tqdm

from pyswip import Prolog


def query_results(p, query):
    # ipdb.set_trace()
    for q in tqdm(query, desc='Proc query'):
        for res in tqdm(p.query(q[0]), desc='Results', leave=False):
            res = {
                k: v.decode() if isinstance(v, (bytes, bytearray)) else v
                for k, v in res.items()
            }
            yield q, res


def clear_str(string):
    # TODO: this breaks negative numbers
    str_map = [
        ['-', '_'],
        [' ', ''],
        [',', ''],
    ]
    string = string.lower()
    for old, new in str_map:
        string = string.replace(old, new)
    return string


def main(args):
    opts = yaml.safe_load(open(args.opts))

    sql_db = opts['sql_addr'] + opts['sql_db']
    tables = {t: pd.read_sql_table(t, sql_db) for t in tqdm(opts['tables'], desc='Getting tables')}

    f = open(opts['sql_db'] + '.pl', 'w')
    prolog = Prolog()
    # pred_map = opts['pred_map']
    for t, df in tqdm(tables.items(), desc='Proc tables'):
        # import ipdb; ipdb.set_trace()
        pred_name = opts['tables'][t].get('pred_map', t)
        # t2 = pred_map.get(t, t)
        # prefixes = opts.get('prefix', {}).get(t, {})
        cols = opts['tables'][t]['cols']
        df = df.loc[:, list(cols)]
        for col, pref in tqdm(cols.items(), desc='Proc cols', leave=False):
            df[col] = df[col].apply(lambda x: pref + str(x))
        for idx, data in tqdm(df.iterrows(), desc='Proc rows', total=len(df), leave=False):
            pred_args = ','.join([clear_str(str(v)) for v in data.values.tolist()])
            to_ass = f'{pred_name}({pred_args})'
            prolog.assertz(to_ass)
            f.write(to_ass + '.\n')

    rules = opts['rules']
    if rules:
        for r in rules:
            prolog.assertz(r)
            f.write(r + '.\n')
    f.close()

    types = opts['types']
    properties = opts['properties']
    connections = opts['connections']

    g = nx.MultiDiGraph()
    print('Getting types')
    # ipdb.set_trace()
    for q, res in query_results(prolog, types):
        # print(res)
        if len(q[1]) == 2:
            node_type = q[1][0]
            node = res[q[1][1]]
        else:
            print(f'Wrong query/triple:', q)

        if node in g:
            # print(f'Node already in graph: {node}')
            pass
        else:
            # g.add_node(node)
            g.add_node(node, nodetype=node_type)

    print('Getting props')
    for q, res in query_results(prolog, properties):
        # print(res)
        if len(q[1]) == 2:
            prop = q[1][0]
            node = res[q[1][1]]
            prop_value = True
        elif len(q[1]) == 3:
            prop = q[1][0]
            node = res[q[1][1]]
            prop_value = res[q[1][2]]
        else:
            print(f'Wrong query/triple:', q)

        prop_type = q[2]

        if node in g:
            props = g.nodes[node].get(prop_type, {})
            if prop_type == 'multi_cat':
                prop_set = props.get(prop, list())
                prop_set.append(prop_value)
                props.update({prop: list(set(prop_set))})
            else:
                if prop_type == 'prop':
                    prop_value = float(prop_value)
                props = g.nodes[node].get(prop_type, {})
                props.update({prop: prop_value})
            g.nodes[node][prop_type] = props
        else:
            print(f'Node not in graph: {node}')
            # g.add_node(node, **{prop: prop_value})

    print('Getting connections')
    for q, res in query_results(prolog, connections):
        # print(res)
        if len(q[1]) == 3:
            edge_label = q[1][0]
            node_1 = res[q[1][1]]
            node_2 = res[q[1][2]]
        else:
            print(f'Wrong query/triple:', q)

        if node_1 not in g:
            print(f'Node not in graph: {node_1}')
            continue
        if node_2 not in g:
            print(f'Node not in graph: {node_2}')
            continue
        g.add_edge(node_1, node_2, label=edge_label)

    # nx.draw_spring(g, with_labels=True)
    # plt.show()

    # import ipdb; ipdb.set_trace()
    # ds_name = osp.basename(opts['sql_db'])[0]
    # json.dump(nx.readwrite.json_graph.cytoscape_data(g), open(ds_name + '.cyjs', 'w'))

    nx.readwrite.gml.write_gml(g, opts['sql_db'] + '.gml')


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--opts')

    ARGS = PARSER.parse_args()
    main(ARGS)
