import json

with open('Glaucoma_All_Models_Colab.py', 'r', encoding='utf-8') as f:
    content = f.read()

cells = []
blocks = content.split('# %%')
for block in blocks:
    if not block.strip():
        continue
    
    lines = block.split('\n')
    if not lines:
        continue
        
    is_markdown = False
    
    if lines[0].strip() == '[markdown]':
        is_markdown = True
        lines = lines[1:]
    
    source = [line + '\n' for line in lines]
    
    # Remove leading blank lines
    while source and source[0].strip() == '':
        source.pop(0)
    
    # Remove trailing blank lines
    while source and source[-1].strip() == '':
        source.pop()
        
    if not source:
        continue

    if is_markdown:
        cleaned_source = []
        for line in source:
            if line.startswith('# '):
                cleaned_source.append(line[2:])
            elif line.startswith('#\n'):
                cleaned_source.append('\n')
            else:
                cleaned_source.append(line)
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": cleaned_source
        })
    else:
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source
        })

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {
            "name": "Glaucoma_All_Models.ipynb"
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('Glaucoma_All_Models_Colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Created Glaucoma_All_Models_Colab.ipynb")
