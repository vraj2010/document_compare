from __future__ import annotations
import difflib

from models import ChunkMatch, ChangeType, Chunk

def _sort_key(item: tuple[ChunkMatch, str]) -> int:
    m, cat = item
    if cat == "Added" and m.chunk_b:
        return getattr(m.chunk_b, 'index_in_doc', 0)
    elif m.chunk_a:
        return getattr(m.chunk_a, 'index_in_doc', 0)
    return 999999

def merge_matches(matches: list[tuple[ChunkMatch, str]]) -> list[tuple[ChunkMatch, str]]:
    """
    Sorts categorized display matches into document order and merges consecutive matches 
    that share the same display category and section_index.
    """
    if not matches:
        return []

    sorted_matches = sorted(matches, key=_sort_key)
    merged: list[tuple[ChunkMatch, str]] = []

    for item in sorted_matches:
        m, cat = item
        if not merged:
            merged.append(item)
            continue
            
        last_m, last_cat = merged[-1]
        
        # Determine if we can merge: contiguous in the original document
        can_merge = False
        if cat == "Added":
            can_merge = (
                last_cat == cat and 
                m.chunk_b and last_m.chunk_b and 
                getattr(m.chunk_b, 'index_in_doc', -1) == getattr(last_m.chunk_b, 'index_in_doc', -2) + 1
            )
        else:
            can_merge = (
                last_cat == cat and 
                m.chunk_a and last_m.chunk_a and 
                getattr(m.chunk_a, 'index_in_doc', -1) == getattr(last_m.chunk_a, 'index_in_doc', -2) + 1
            )
            
        if can_merge:
            # Create a new ChunkMatch combining texts
            # We instantiate new chunks to avoid mutating the original report data
            
            new_chunk_a = Chunk(**last_m.chunk_a.model_dump()) if last_m.chunk_a else None
            new_chunk_b = Chunk(**last_m.chunk_b.model_dump()) if last_m.chunk_b else None
            
            if m.chunk_a and new_chunk_a:
                new_chunk_a.text += "\n\n" + m.chunk_a.text
            elif m.chunk_a and not new_chunk_a:
                new_chunk_a = Chunk(**m.chunk_a.model_dump())
                
            if m.chunk_b and new_chunk_b:
                new_chunk_b.text += "\n\n" + m.chunk_b.text
            elif m.chunk_b and not new_chunk_b:
                new_chunk_b = Chunk(**m.chunk_b.model_dump())
                
            # Combine critical info changes
            combined_crit = list(last_m.critical_info_changes)
            combined_crit.extend(m.critical_info_changes)
                
            new_m = ChunkMatch(
                chunk_a=new_chunk_a if new_chunk_a else m.chunk_a,
                chunk_b=new_chunk_b,
                change_type=last_m.change_type,
                similarity_score=last_m.similarity_score,
                fuzzy_score=last_m.fuzzy_score,
                semantic_score=last_m.semantic_score,
                semantic_analysis=last_m.semantic_analysis,
                critical_info_changes=combined_crit
            )
            merged[-1] = (new_m, last_cat)
        else:
            merged.append(item)
            
    return merged

import re

def highlight_diff(text_a: str, text_b: str) -> tuple[str, str]:
    """
    Highlights differences between two strings using difflib, preserving whitespaces/newlines.
    Returns (html_a, html_b).
    """
    if not text_a or not text_b:
        return text_a, text_b
        
    # Split keeping whitespaces so we can reconstruct exactly
    tokens_a = [t for t in re.split(r'(\s+)', text_a) if t]
    tokens_b = [t for t in re.split(r'(\s+)', text_b) if t]
    
    matcher = difflib.SequenceMatcher(None, tokens_a, tokens_b)
    out_a = []
    out_b = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        chunk_a = "".join(tokens_a[i1:i2])
        chunk_b = "".join(tokens_b[j1:j2])
        
        # Don't wrap pure whitespace in tags
        def _wrap(chunk, tag_fmt):
            if not chunk: return ""
            if chunk.isspace(): return chunk
            return tag_fmt.format(chunk)
            
        del_fmt = '<span style="background-color: #ffcdd2; color: #b71c1c;">{}</span>'
        ins_fmt = '<span style="background-color: #c8e6c9; color: #1b5e20;">{}</span>'
        
        if tag == 'equal':
            out_a.append(chunk_a)
            out_b.append(chunk_b)
        elif tag == 'delete':
            out_a.append(_wrap(chunk_a, del_fmt))
        elif tag == 'insert':
            out_b.append(_wrap(chunk_b, ins_fmt))
        elif tag == 'replace':
            out_a.append(_wrap(chunk_a, del_fmt))
            out_b.append(_wrap(chunk_b, ins_fmt))
            
    return "".join(out_a), "".join(out_b)
