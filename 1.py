def find_s(examples):
    # Get number of attributes
    hypothesis = ['0'] * (len(examples[0]) - 1)

    for row in examples:
        if row[-1] == "Yes":  # only consider positive examples
            for i in range(len(hypothesis)):
                if hypothesis[i] == '0':
                    hypothesis[i] = row[i]
                elif hypothesis[i] != row[i]:
                    hypothesis[i] = '?'
    return hypothesis

def candidate_elimination(examples):
    # initialize S (most specific) and G (most general)
    num_attributes = len(examples[0]) - 1
    S = ['0'] * num_attributes
    G = [['?'] * num_attributes]

    for row in examples:
        attrs, target = row[:-1], row[-1]

        if target == "Yes":  # Positive Example
            # generalize S
            for i in range(num_attributes):
                if S[i] == '0':
                    S[i] = attrs[i]
                elif S[i] != attrs[i]:
                    S[i] = '?'

            # remove inconsistent hypotheses from G
            G = [g for g in G if all(g[i] == '?' or g[i] == attrs[i] or S[i] == '?' for i in range(num_attributes))]

        else:  # Negative Example
            # specialize G
            new_G = []
            for g in G:
                for i in range(num_attributes):
                    if g[i] == '?':
                        if S[i] != '?':
                            g_new = g.copy()
                            g_new[i] = S[i]
                            new_G.append(g_new)
            G.extend(new_G)
            G = [g for g in G if any(g[i] == '?' or g[i] == S[i] for i in range(num_attributes))]

    return S, G
examples = [
    ["Young", "High", "Male", "Yes"],
    ["Young", "High", "Female", "Yes"],
    ["Middle", "High", "Male", "No"],
    ["Old", "Low", "Male", "No"],
    ["Young", "Low", "Male", "Yes"]
]
print("Find-S Hypothesis:", find_s(examples))
S, G = candidate_elimination(examples)
print("Candidate Elimination Specific Hypothesis (S):", S)
print("Candidate Elimination General Hypotheses (G):", G)
