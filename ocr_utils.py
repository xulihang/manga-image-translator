import unicodedata
from utils import BBox, Quadrilateral, image_resize, quadrilateral_can_merge_region
from typing import List, Tuple
import networkx as nx
import itertools
from collections import Counter
import numpy as np
import cv2

def resize_keep_aspect(img, size) :
    ratio = (float(size)/max(img.shape[0], img.shape[1]))
    new_width = round(img.shape[1] * ratio)
    new_height = round(img.shape[0] * ratio)
    return cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_LINEAR_EXACT)
    
def overlay_mask(img, mask) :
    img2 = img.copy().astype(np.float32)
    mask_fp32 = (mask > 10).astype(np.uint8) * 2
    mask_fp32[mask_fp32 == 0] = 1
    mask_fp32 = mask_fp32.astype(np.float32) * 0.5
    img2 = img2 * mask_fp32[:, :, None]
    return img2.astype(np.uint8)
    
def _is_whitespace(ch):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if ch == " " or ch == "\t" or ch == "\n" or ch == "\r" or ord(ch) == 0:
        return True
    cat = unicodedata.category(ch)
    if cat == "Zs":
        return True
    return False


def _is_control(ch):
    """Checks whether `chars` is a control character."""
    """Checks whether `chars` is a whitespace character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if ch == "\t" or ch == "\n" or ch == "\r":
        return False
    cat = unicodedata.category(ch)
    if cat in ("Cc", "Cf"):
        return True
    return False


def _is_punctuation(ch):
    """Checks whether `chars` is a punctuation character."""
    """Checks whether `chars` is a whitespace character."""
    cp = ord(ch)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
        (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(ch)
    if cat.startswith("P"):
        return True
    return False
    

def count_valuable_text(text) :
    return sum([1 for ch in text if not _is_punctuation(ch) and not _is_control(ch) and not _is_whitespace(ch)])
    
def generate_text_direction(bboxes: List[Quadrilateral]) :
    G = nx.Graph()
    for i, box in enumerate(bboxes) :
        G.add_node(i, box = box)
    for ((u, ubox), (v, vbox)) in itertools.combinations(enumerate(bboxes), 2) :
        if quadrilateral_can_merge_region(ubox, vbox) :
            G.add_edge(u, v)
    for node_set in nx.algorithms.components.connected_components(G) :
        nodes = list(node_set)
        # majority vote for direction
        dirs = [box.direction for box in [bboxes[i] for i in nodes]]
        majority_dir = Counter(dirs).most_common(1)[0][0]
        # sort
        if majority_dir == 'h' :
            nodes = sorted(nodes, key = lambda x: bboxes[x].aabb.y + bboxes[x].aabb.h // 2)
        elif majority_dir == 'v' :
            nodes = sorted(nodes, key = lambda x: -(bboxes[x].aabb.x + bboxes[x].aabb.w))
        # yield overall bbox and sorted indices
        for node in nodes :
            yield bboxes[node], majority_dir
            
def merge_bboxes_text_region(bboxes: List[Quadrilateral]) :
    G = nx.Graph()
    for i, box in enumerate(bboxes) :
        G.add_node(i, box = box)
    for ((u, ubox), (v, vbox)) in itertools.combinations(enumerate(bboxes), 2) :
        if quadrilateral_can_merge_region(ubox, vbox) :
            G.add_edge(u, v)
    for node_set in nx.algorithms.components.connected_components(G) :
        nodes = list(node_set)
        # get overall bbox
        txtlns = np.array(bboxes)[nodes]
        kq = np.concatenate([x.pts for x in txtlns], axis = 0)
        if sum([int(a.is_approximate_axis_aligned) for a in txtlns]) > len(txtlns) // 2 :
            max_coord = np.max(kq, axis = 0)
            min_coord = np.min(kq, axis = 0)
            merged_box = np.maximum(np.array([
                np.array([min_coord[0], min_coord[1]]),
                np.array([max_coord[0], min_coord[1]]),
                np.array([max_coord[0], max_coord[1]]),
                np.array([min_coord[0], max_coord[1]])
                ]), 0)
            bbox = np.concatenate([a[None, :] for a in merged_box], axis = 0).astype(int)
        else :
            # TODO: use better method
            bbox = np.concatenate([a[None, :] for a in get_mini_boxes(kq)], axis = 0).astype(int)
        # calculate average fg and bg color
        fg_r = round(np.mean([box.fg_r for box in [bboxes[i] for i in nodes]]))
        fg_g = round(np.mean([box.fg_g for box in [bboxes[i] for i in nodes]]))
        fg_b = round(np.mean([box.fg_b for box in [bboxes[i] for i in nodes]]))
        bg_r = round(np.mean([box.bg_r for box in [bboxes[i] for i in nodes]]))
        bg_g = round(np.mean([box.bg_g for box in [bboxes[i] for i in nodes]]))
        bg_b = round(np.mean([box.bg_b for box in [bboxes[i] for i in nodes]]))
        # majority vote for direction
        dirs = [box.direction for box in [bboxes[i] for i in nodes]]
        majority_dir = Counter(dirs).most_common(1)[0][0]
        # sort
        if majority_dir == 'h' :
            nodes = sorted(nodes, key = lambda x: bboxes[x].aabb.y + bboxes[x].aabb.h // 2)
        elif majority_dir == 'v' :
            nodes = sorted(nodes, key = lambda x: -(bboxes[x].aabb.x + bboxes[x].aabb.w))
        # yield overall bbox and sorted indices
        yield bbox, nodes, majority_dir, fg_r, fg_g, fg_b, bg_r, bg_g, bg_b
        
            