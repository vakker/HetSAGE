import argparse
from os import path as osp

import networkx as nx
import pandas as pd
import yaml
from pyswip import Prolog
from tqdm import tqdm


def single_query(p, query):
    for res in tqdm(p.query(query), desc='Results', leave=False):
        res = {k: v.decode() if isinstance(v, (bytes, bytearray)) else v for k, v in res.items()}
        yield res


def query_results(p, query):
    for q in tqdm(query, desc='Proc query'):
        for res in single_query(p, q[0]):
            yield q, res
        # for res in tqdm(p.query(q[0]), desc='Results', leave=False):
        #     res = {
        #         k: v.decode() if isinstance(v, (bytes, bytearray)) else v
        #         for k, v in res.items()
        #     }
        #     yield q, res


def clear_str(string):
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
    target_dir = osp.dirname(args.opts)

    sql_db = opts['sql_addr'] + opts['sql_db']
    tables = {t: pd.read_sql_table(t, sql_db) for t in tqdm(opts['tables'], desc='Getting tables')}

    f = open(osp.join(target_dir, 'bk.pl'), 'w')
    prolog = Prolog()
    for t, df in tqdm(tables.items(), desc='Proc tables'):
        pred_name = opts['tables'][t].get('pred_map', t)
        cols = opts['tables'][t]['cols']
        df = df.loc[:, list(cols)]
        for col, pref in tqdm(cols.items(), desc='Proc cols', leave=False):
            if isinstance(pref, str):
                df[col] = df[col].apply(lambda x: clear_str(pref + str(x)))
        for idx, data in tqdm(df.iterrows(), desc='Proc rows', total=len(df), leave=False):
            pred_args = ','.join([str(v) for v in data.values.tolist()])
            to_ass = f'{pred_name}({pred_args})'
            prolog.assertz(to_ass)
            f.write(to_ass + '.\n')

    rules = opts['rules']
    if rules:
        for r in rules:
            prolog.assertz(r)
            f.write(r + '.\n')
    f.close()

    rules = opts['aleph']
    for r in rules:
        prolog.assertz(r)
    # prolog.assertz(opts['aleph']['pos'])
    # prolog.assertz(opts['aleph']['neg'])

    for s in ['pos', 'neg']:
        with open(osp.join(target_dir, s + '.pl'), 'w') as f:
            for res in single_query(prolog, s + '(X)'):
                sample = res['X']
                to_ass = f'target({sample})'
                f.write(to_ass + '.\n')

    types = opts['types']
    properties = opts['properties']
    connections = opts['connections']

    g = nx.MultiDiGraph()
    print('Getting types')
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
            print(f'Node not in graph: {node}, {prop}')

    print('Getting connections')
    for q, res in query_results(prolog, connections):
        if len(q[1]) == 3:
            edge_label = q[1][0]
            node_1 = res[q[1][1]]
            node_2 = res[q[1][2]]
        else:
            print(f'Wrong query/triple:', q)

        if node_1 not in g:
            print(f'Node not in graph: {node_1} for {node_1} {edge_label} {node_2}')
            continue
        if node_2 not in g:
            print(f'Node not in graph: {node_2} for {node_1} {edge_label} {node_2}')
            continue
        g.add_edge(node_1, node_2, label=edge_label)

    nx.write_gpickle(g, osp.join(target_dir, "graph.gpickle"))
    if args.gml:
        nx.readwrite.gml.write_gml(g, osp.join(target_dir, 'graph.gml'))


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--opts')
    PARSER.add_argument('--gml', action='store_true')

    ARGS = PARSER.parse_args()
    main(ARGS)
