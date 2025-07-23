if __name__ == "__main__":

    import json

    with open("eigenvalue_data_cleaned.json", "r") as f:
        data = json.load(f)

    merged_data = []
    for entry in data:
        eigs = entry["eigenvalues"]
        probs = entry["probabilities"]

        merged_eigs = []
        merged_probs = []
        zero_prob_total = 0.0

        for eig, prob in zip(eigs, probs):
            if eig == 0.0:
                zero_prob_total += prob
            else:
                merged_eigs.append(eig)
                merged_probs.append(prob)

        if zero_prob_total > 0:
            merged_eigs.append(0.0)
            merged_probs.append(zero_prob_total)

        merged_data.append({
            "eigenvalues": merged_eigs,
            "probabilities": merged_probs
        })

    with open("eigenvalue_data_merged.json", "w") as f:
        json.dump(merged_data, f, indent=2)
    # import json
    #
    # THRESH_PROB = 1e-5
    # THRESH_EIG = 1e-5
    #
    # with open("eigenvalue_data.json", "r") as f:
    #     data = json.load(f)
    #
    # cleaned_data = []
    # for entry in data:
    #     eigs = entry["eigenvalues"]
    #     probs = entry["probabilities"]
    #
    #     # Apply threshold filters
    #     new_eigs = []
    #     new_probs = []
    #
    #     for eig, prob in zip(eigs, probs):
    #         if prob >= THRESH_PROB:
    #             if abs(eig) < THRESH_EIG:
    #                 eig = 0.0
    #             new_eigs.append(eig)
    #             new_probs.append(prob)
    #
    #     cleaned_data.append({
    #         "eigenvalues": new_eigs,
    #         "probabilities": new_probs
    #     })
    #
    # with open("eigenvalue_data_cleaned.json", "w") as f:
    #     json.dump(cleaned_data, f, indent=2)
    #
    #     

