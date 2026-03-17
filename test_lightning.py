#!/usr/bin/env python
"""Test the QuantumClassifier"""
import sys
sys.path.insert(0, 'e:/NEWSCAT')

try:
    from backend.models.lightning_classifier import QuantumClassifier
    clf = QuantumClassifier()
    result = clf.classify('Apple reported record quarterly earnings today, beating analyst expectations.')
    print("Category:", result.get('category'))
    print("Confidence:", result.get('confidence'))
    print("Top predictions:", result.get('top_predictions'))
    print("Method:", result.get('method'))
except Exception as e:
    print("Error loading QuantumClassifier:", str(e))
    import traceback
    traceback.print_exc()
