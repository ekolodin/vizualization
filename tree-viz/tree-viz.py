import sys
import argparse
import networkx as nx
import matplotlib.pyplot as plt


class TreeViz:

    def __init__(self):
        self._counter = 0

    def __inorder(self, tree, points, node, depth):
        points.append(node)

        if tree.get(node) and len(tree[node]['nodes']) >= 1:
            points, tree = self.__inorder(tree, points, tree[node]['nodes'][0], depth + 1)

        points.append(node)
        tree[node]['x'] = self._counter
        tree[node]['y'] = -depth
        self._counter += 1

        if tree.get(node) and len(tree[node]['nodes']) == 2:
            points, tree = self.__inorder(tree, points, tree[node]['nodes'][1], depth + 1)

        return points, tree

    def __compress(self, tree, node, depth):
        left, right = 0, 0

        if tree.get(node) and len(tree[node]['nodes']) >= 1:
            left = tree[node]['nodes'][0]
            tree = self.__compress(tree, left, depth + 1)

        if tree.get(node) and len(tree[node]['nodes']) == 2:
            right = tree[node]['nodes'][1]
            tree = self.__compress(tree, right, depth + 1)

        left_contour, right_contour = tree[left]['contour'], tree[right]['contour']

        depths = list(set(list(left_contour.keys()) + list(right_contour.keys())))

        gap = int(1e9)
        for d in depths:

            if not left_contour.get(d) or not right_contour.get(d):
                continue

            max_left = tree[max(left_contour[d], key=lambda x: tree[x]['x'])]['x']
            min_right = tree[min(right_contour[d], key=lambda x: tree[x]['x'])]['x']
            gap = min(gap, min_right - 1 - max_left)

        for d in depths:
            for x in right_contour[d] if right_contour.get(d) else []:
                if gap != sys.maxsize:
                    tree[x]['x'] -= gap

        for d in depths:
            tree[node]['contour'][d] = list(set(
                (left_contour[d] if left_contour.get(d) else []) + (right_contour[d] if right_contour.get(d) else [])
            ))

        tree[node]['x'] = tree[left]['x']
        tree[node]['contour'][depth] = [node]

        return tree

    def __call__(self, tree):
        points = []

        points, tree = self.__inorder(tree, points, 0, 0)
        tree = self.__compress(tree, 0, 0)

        plt.figure(figsize=(16, 8))

        prev = 0
        for point in points:
            first, second = tree[prev], tree[point]
            if second['y'] < first['y']:
                plt.plot([second['x'], first['x']], [second['y'], first['y']], 'o-')
            prev = point
        plt.show()

        self._counter = 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, default='tree-84n.xml', help='path to tree file')

    args = parser.parse_args()

    graphml = nx.read_graphml(args.path)

    node_numeration = {}
    for i, (node, _) in enumerate(graphml.nodes(data=True)):
        node_numeration[node] = i

    graph = {}

    for from_, to_, _ in graphml.edges(data=True):
        from_, to_ = node_numeration[from_], node_numeration[to_]

        if graph.get(from_):
            graph[from_]['nodes'].append(to_)
        else:
            graph[from_] = {'nodes': [to_], 'x': 0, 'y': 0, 'contour': {}}

    for node, _ in graphml.nodes(data=True):
        node = node_numeration[node]
        if not graph.get(node):
            graph[node] = {'nodes': [], 'x': 0, 'y': 0, 'contour': {}}

    TreeViz()(graph)
