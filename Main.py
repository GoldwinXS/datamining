import requests
import bs4
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span
import pandas as pd
import networkx as nx
import plotly.graph_objs as go
import matplotlib.pyplot as plt


target_url = 'https://en.wikipedia.org/wiki/Distributed_computing'

# get website html
req = requests.get(target_url)

# parse text with bs4
soup = bs4.BeautifulSoup(req.text, 'html.parser')

# find all of the p tags
tags = soup.find_all('p')

# join all of the text from those tags into one long string
text = ' '.join([tag.text.replace('\n', '') for tag in tags])

# separate the string at every period
texts = text.split('.')

# load SpaCy model. Used in many functions below. 
nlp = spacy.load('en_core_web_sm')


def get_entities(sent):
    """get entities function, an example taken from the internet with the explanation"""

    """I have defined a few empty variables in this chunk. prv_tok_dep and prv_tok_text will hold the dependency
    tag of the previous word in the sentence and that previous word itself, respectively. prefix and modifier will
    hold the text that is associated with the subject or the object."""

    ent1 = ""
    ent2 = ""

    prv_tok_dep = ""  # dependency tag of previous token in the sentence
    prv_tok_text = ""  # previous token in the sentence

    prefix = ""
    modifier = ""

    #############################################################

    for tok in nlp(sent):

        """Next, we will loop through the tokens in the sentence. We will first check if the token is a punctuation
        mark or not. If yes, then we will ignore it and move on to the next token. If the token is a part of a
        compound word (dependency tag = “compound”), we will keep it in the prefix variable. A compound word is a 
        combination of multiple words linked to form a word with a new meaning 
        (example – “Football Stadium”, “animal lover”).

        As and when we come across a subject or an object in the sentence, we will add this prefix to it. 
        We will do the same thing with the modifier words, such as “nice shirt”, “big house”, etc."""

        # if token is a punctuation mark then move on to the next token
        if tok.dep_ != "punct":
            # check: token is a compound word or not
            if tok.dep_ == "compound":
                prefix = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    prefix = prv_tok_text + " " + tok.text

            # check: token is a modifier or not
            if tok.dep_.endswith("mod") == True:
                modifier = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    modifier = prv_tok_text + " " + tok.text

            """Here, if the token is the subject, then it will be captured as the first entity in the ent1 variable. 
            Variables such as prefix, modifier, prv_tok_dep, and prv_tok_text will be reset."""
            if tok.dep_.find("subj") == True:
                ent1 = modifier + " " + prefix + " " + tok.text
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
                prv_tok_text = ""

                """Here, if the token is the object, then it will be captured as the second entity 
                in the ent2 variable. Variables such as prefix, modifier, prv_tok_dep, 
                and prv_tok_text will again be reset."""

            if tok.dep_.find("obj") == True:
                ent2 = modifier + " " + prefix + " " + tok.text

            """Once we have captured the subject and the object in the sentence, we will update the previous 
            token and its dependency tag."""

            # update variables
            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text
    #############################################################

    return [ent1.strip(), ent2.strip()]


def find_relations(docs):
    # Matcher class object
    matcher = Matcher(nlp.vocab)

    # define the pattern
    pattern = [{'DEP': 'ROOT'},
               {'DEP': 'prep', 'OP': "?"},
               {'DEP': 'agent', 'OP': "?"},
               {'POS': 'ADJ', 'OP': "?"}]

    matcher.add("matching_1", None, pattern)

    matches = matcher(docs)
    k = len(matches) - 1

    try:
        span = docs[matches[k][1]:matches[k][2]]
        return (span.text)

    except IndexError:
        pass


def get_all_text_relations(list_of_str):
    relation = [find_relations(nlp(t)) for t in list_of_str]

    entities = [get_entities(t) for t in list_of_str]

    source = [r[0] for r in entities]
    target = [r[1] for r in entities]

    graph_data = {'target': target,
                  'edge': relation,
                  'source': source}

    return pd.DataFrame(graph_data)


# create knowledge graph
kg = get_all_text_relations(texts)


def plot_kg(kg_df):
    """ Plot a knowledge graph from a pandas dataframe, using the from_pandas_edgelist function in networkx"""

    # create the graph
    G = nx.from_pandas_edgelist(kg_df, "source", "target", edge_attr=True, create_using=nx.MultiDiGraph())
    plt.figure(figsize=(12, 12))

    # arrange pos of all nodes
    pos = nx.spring_layout(G, k=0.5)  # k regulates the distance between nodes

    # find how many adjacencies each node has in a dict
    adjs = {node: len(num) for node, num in G.adjacency()}
    max_adj = max(adjs.values())

    # prepare nodes
    node_x = []
    node_y = []
    node_z = []
    names = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(adjs[node])
        names.append(node)

    node_trace = go.Scatter3d(
        text=names,
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(
            size=node_z,
            color=node_z,  # set color to an array/list of desired values
            colorscale='Viridis',
            opacity=0.8)
    )

    node_adjacencies = []
    node_text = []
    for i, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f'{names[i]} connections: ' + str(len(adjacencies[1])))

    # prepare edges
    edges = []

    edge_x = []
    edge_y = []
    for s, t in G.edges():
        # add two edges
        x0, y0 = pos[s]
        x1, y1 = pos[t]
        z0, z1 = adjs[s], adjs[t]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)

        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

        edge_y.append(z0)
        edge_y.append(z1)
        edge_y.append(None)

        edge_trace = go.Scatter3d(x=[x0, x1], y=[y0, y1], z=[z0, z1],
                                  line=dict(width=(adjs[s] / max_adj) + 1, color='#000000'),
                                  hoverinfo='none',
                                  mode='lines')
        edges.append(edge_trace)

    # color nodes pos
    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    # define final plot
    fig = go.Figure(data=[node_trace] + edges,
                    layout=go.Layout(
                        title='<br>Network graph made with Python',
                        titlefont_size=16,
                        showlegend=False,
                        margin=dict(b=20, l=5, r=5, t=40)))

    fig.show()


plot_kg(kg)


