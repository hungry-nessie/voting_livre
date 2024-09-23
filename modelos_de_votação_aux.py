import pandas as pd # type: ignore
import random
import numpy as np # type: ignore
from collections import defaultdict
import copy
import matplotlib.pyplot as plt # type: ignore
from scipy.interpolate import make_interp_spline # type: ignore
from scipy.stats import spearmanr 

candidate_names = [
    "Maçã", "Banana", "Cenoura", "Tâmara", "Berinjela", "Figo", "Uva", "Melão",
    "Alface Iceberg", "Jaca", "Kiwi", "Limão", "Manga", "Nectarina",
    "Laranja", "Mamão", "Marmelo", "Framboesa", "Morango", "Tomate",
    "Fruta Ugli", "Baunilha", "Melancia", "Xigua", "Pimentão Amarelo",
    "Abobrinha", "Alcachofra", "Brócolis", "Repolho", "Daikon", "Endívia",
    "Funcho", "Alho", "Raiz-forte", "Figo-da-Índia", "Jalapeño", "Couve",
    "Alho-poró", "Cogumelo", "Acelga", "Quiabo", "Pastinaca", "Quinoa",
    "Rabanete", "Espinafre", "Nabo", "Inhame Roxo", "Cebola Vidalia", "Agrião",
    "Inhame", "Flor de Abobrinha", "Abóbora Bolota", "Pimentão", "Melão Cantaloupe",
    "Pitaya", "Escarola", "Feto Fiddlehead", "Gengibre", "Habanero",
    "Planta de Gelo", "Jicama", "Couve-rábano", "Lichia", "Amora",
    "Laranja-de-Umbigo", "Azeitona", "Abacaxi", "Alface Romana", "Azedinha",
    "Tangerina", "Feijão Urad", "Laranja Valencia", "Aspargo Branco", "Yuzu",
    "Ziziphus", "Damasco", "Amora", "Pepino", "Durian", "Elderberry",
    "Feijoa", "Groselha", "Huckleberry", "Imbe", "Jujuba", "Kumquat",
    "Longan", "Mandarina", "Nance", "Orégano", "Pêssego", "Rambutan",
    "Graviola", "Tamarindo", "Melão Ugli", "Baunilha", "Maçã-de-cera", "Yacon",
    "Uva Zinfandel", "Broto de Alfafa", "Couve de Bruxelas", "Couve-flor",
    "Abóbora Delicata", "Alho Elefante", "Frisée", "Feijão Verde",
    "Palmito", "Alface", "Alcachofra de Jerusalém", "Feijão", "Feijão Lima",
    "Feijão Mungo", "Noni", "Cebola", "Romã", "Radicchio", "Sálvia",
    "Acelga", "Tomatillo", "Ume", "Vidalia"
]



################################################################################
#                              CREATING CONTEXT                                #
################################################################################

def generate_probs_for_internal_voters(no_of_candidates, scenario, noise_level=0.1):
    if scenario == "uniform":
        probabilities = [1/no_of_candidates] * no_of_candidates
    elif scenario == "moderate_bias":
        probabilities = [1/no_of_candidates] * no_of_candidates
        for i in range(int(no_of_candidates * 0.3)):
            probabilities[i] *= 5/no_of_candidates
    else:
        probabilities = [1] * 3 + [0.03] * (no_of_candidates - 3)

    total = sum(probabilities)
    probabilities = [p/total for p in probabilities]

    noisy_probabilities = []
    for p in probabilities:
        noise = random.gauss(0, noise_level)  # Using Gaussian noise
        noisy_p = max(p + noise, 0)  # Ensure non-negative
        noisy_probabilities.append(noisy_p)

    total_noisy = sum(noisy_probabilities)
    normalized_noisy_probabilities = [p/total_noisy for p in noisy_probabilities]

    return normalized_noisy_probabilities

def generate_probs_for_external_voters(internal_probabilities, internal_external_voting_disparity):
    """
    Generate probabilities for external voters based on internal probabilities and the level of disparity between internal and external voters.

    Parameters
    ----------
    internal_probabilities : list of float
        The probabilities of the internal voters
    internal_external_voting_disparity : str
        The level of disparity between internal and external voters. Must be one of "slightly_different", "somewhat_different", or "very_different"

    Returns
    -------
    list
        A list of the final probabilities for the external voters
    """
    external_probabilities = internal_probabilities.copy()
    variation_factor = random.uniform(0.2, 0.4)  # Random variation factor

    if internal_external_voting_disparity == "slightly_different":
        for i in range(len(external_probabilities)):
            adjustment = random.uniform(-variation_factor, variation_factor)
            external_probabilities[i] += adjustment * internal_probabilities[i]
    elif internal_external_voting_disparity == "somewhat_different":
        for i in range(len(external_probabilities)):
            if random.random() < 0.7:  # Increase the likelihood of a drastic change
                external_probabilities[i] = random.uniform(0.01, 0.1) if random.random() < 0.5 else random.uniform(0.9, 1)
            else:
                adjustment = random.uniform(-2 * variation_factor, 2 * variation_factor)
                external_probabilities[i] += adjustment * internal_probabilities[i]
    elif internal_external_voting_disparity == "very_different":
        total_internal = sum(internal_probabilities)
        max_internal = max(internal_probabilities)
        min_internal = min(internal_probabilities)
        for i in range(len(external_probabilities)):
            # Oppose the probabilities by inverting them relative to the range
            external_probabilities[i] = max_internal + min_internal - internal_probabilities[i]
    else:
        raise ValueError("Invalid difference level")

    external_probabilities = [max(p, 0) for p in external_probabilities]

    total = sum(external_probabilities)
    normalized_external_probabilities = [p/total for p in external_probabilities]

    return normalized_external_probabilities



def create_context(no_of_candidates, scenario, internal_external_voting_disparity):
    """
    Create a context for the voting simulation, given the number of candidates, the scenario, and the level of disparity between internal and external voters.

    Parameters
    ----------
    no_of_candidates : int
        The number of candidates in the election
    scenario : str
        The scenario to generate probabilities for. Must be one of "uniform", "moderate_bias", or "strong_bias"
    internal_external_voting_disparity : str
        Level of disparity between internal and external voters. Must be one of
        "slightly_different", "somewhat_different", or "very_different"

    Returns
    -------
    list
        A list of dictionaries, where each dict has the keys 'name', 'internal_probability', and
        'external_probability' with the name of the candidate, the probability of the internal voters, and
        the probability of the external voters, respectively.
    """
    
    internal_probs = generate_probs_for_internal_voters(no_of_candidates, scenario)
    external_probs = generate_probs_for_external_voters(internal_probs, internal_external_voting_disparity)

  # Creating a list of dictionaries
    context = []
    for i in range(no_of_candidates):
        candidate_info = {
            "name": candidate_names[i],
            "internal_probability": internal_probs[i],
            "external_probability": external_probs[i]
        }
        context.append(candidate_info)

    return context


point_system = [10, 6.67, 4.44, 2.96, 1.98, 1.32]



################################################################################
#                                 VOTING DAY                                   #
################################################################################
def vote(population_size, voters_distribution, context):
    # Determine the number of internal and external voters based on distribution
    """
    Simulate a vote with internal and external voters, given the population size, distribution of voters, and context.

    Parameters
    ----------
    population_size : int
        The size of the population
        The distribution of voters. Must be one of "equal", "more_internal_voters", or "more_external_voters"
    context : list of dict
        The context list of dictionaries, where each dict has the keys 'name', 'internal_probability', and
        'external_probability' with the name of the candidate, the probability of the internal voters, and
        the probability of the external voters, respectively.

    Returns
    -------
    tuple
        A tuple containing the internal and external voting intentions, and the updated context
    """
    voters_distribution : str

    if voters_distribution == "equal":
        internal_voters = int(population_size * 0.5)
        external_voters = int(population_size * 0.5)
    elif voters_distribution == "more_internal_voters":
        internal_voters = int(population_size * 0.7)
        external_voters = int(population_size * 0.3)
    elif voters_distribution == "more_external_voters":
        internal_voters = int(population_size * 0.3)
        external_voters = int(population_size * 0.7)

    # Initialize lists for voting intention
    internal_voting_intention = []
    external_voting_intention = []

    # Reset internal and external votes in context
    for candidate in context:
        candidate["internal_votes"] = 0
        candidate["internal_candidate_rank"] = 0
        candidate["external_votes"] = 0
        candidate["external_candidate_rank"] = 0
        candidate["no_rules_candidate_score"] = 0
        candidate["no_rules_candidate_rank"] = 0

    # Internal voters' weighted votes (ordered lists of 4 to 6 candidates)
    internal_probabilities = [c["internal_probability"] for c in context]
    for _ in range(internal_voters):
        num_candidates_to_vote = random.randint(4, 6)
        vote_indices = random.choices(range(len(context)), internal_probabilities, k=num_candidates_to_vote)
        internal_voting_intention.append(vote_indices)

        for idx, candidate_index in enumerate(vote_indices):
            context[candidate_index]["internal_votes"] += point_system[idx]

    # Sort candidates by their total score to determine ranking
    score_sorted = sorted(context, key=lambda x: x["internal_votes"], reverse=True)
    for rank, candidate in enumerate(score_sorted, start=1):
        candidate["internal_candidate_rank"] = rank

    # External voters' single votes (one candidate per voter)
    external_probabilities = [c["external_probability"] for c in context]
    for _ in range(external_voters):
        candidate_index = random.choices(range(len(context)), external_probabilities, k=1)[0]
        external_voting_intention.append(candidate_index)

        context[candidate_index]["external_votes"] += point_system[0]

    # Sort candidates by their total score to determine ranking
    score_sorted = sorted(context, key=lambda x: x["external_votes"], reverse=True)
    for rank, candidate in enumerate(score_sorted, start=1):
        candidate["external_candidate_rank"] = rank

    # Combine internal and external votes into candidate scores
    for candidate in context:
        candidate["no_rules_candidate_score"] = candidate["internal_votes"] + candidate["external_votes"]

    # Sort candidates by their total score to determine ranking
    score_sorted = sorted(context, key=lambda x: x["no_rules_candidate_score"], reverse=True)
    for rank, candidate in enumerate(score_sorted, start=1):
        candidate["no_rules_candidate_rank"] = rank

    return context

# Because we want to make the code a bit more digestable, we will encapsulate parts of it
def create_voting_results(no_of_candidates, 
                          scenario, 
                          internal_external_voting_disparity, 
                          population_size, 
                          voters_distribution):
    """
    Simulate a vote with given parameters.

    Parameters
    ----------
    no_of_candidates : int
        The number of candidates in the election
    scenario : str
        The scenario to generate probabilities for. Must be one of "uniform", "moderate_bias", or "strong_bias"
    internal_external_voting_disparity : str
        Level of disparity between internal and external voters. Must be one of
        "slightly_different", "somewhat_different", or "very_different"
    population_size : int
        The size of the population
    voters_distribution : tuple
        A tuple of two numbers, the first is the number of internal voters, and the second is the number of external voters

    Returns
    -------
    list
        A list of dictionaries, where each dict has the keys 'name', 'internal_votes', 'external_votes', 'internal_candidate_rank', 'external_candidate_rank', 'no_rules_candidate_score', 'no_rules_candidate_rank' with the name of the candidate, the number of internal and external votes for that candidate, the rank of that candidate in the internal and external votes, and the rank of that candidate with no rules applied, respectively.
    """
    context = create_context(no_of_candidates, scenario, internal_external_voting_disparity)
    voting_results = vote(population_size, voters_distribution = voters_distribution, context=context)

    return context


################################################################################
#              ELECTORAL RESULTS & POST-ELECTION SATISFACTION                  #
################################################################################
def process_votes_internal_gets_2_points(context):
    """
    Process the votes in the context by giving internal votes twice the points.

    Parameters
    ----------
    context : list of dict
        The context list of dictionaries, where each dict has the keys 'internal_votes'
        and 'external_votes' with the number of internal and external votes for that
        candidate.

    Returns
    -------
    list
        A list of the final scores for each candidate, with the internal votes
        doubled and added to the external votes.
    """
    internal_votes = [(item['internal_votes'] * 2) for item in context]
    external_votes = [item['external_votes'] for item in context]
    final_scores = [internal_votes[i] + external_votes[i] for i in range(len(internal_votes))]
    return final_scores

def process_votes_internal_gets_3_points(context):
    """
    Process the votes in the context by giving internal votes twice the points.

    Parameters
    ----------
    context : list of dict
        The context list of dictionaries, where each dict has the keys 'internal_votes'
        and 'external_votes' with the number of internal and external votes for that
        candidate.

    Returns
    -------
    list
        A list of the final scores for each candidate, with the internal votes
        tripled and added to the external votes.
    """
    internal_votes = [(item['internal_votes'] * 3) for item in context]
    external_votes = [item['external_votes'] for item in context]
    final_scores = [internal_votes[i] + external_votes[i] for i in range(len(internal_votes))]
    return final_scores

def process_votes_external_vote_cap_80(context):
    """
    Process the votes in the context by capping the external votes to 70% of the total.

    Parameters
    ----------
    context : list of dict
        The context list of dictionaries, where each dict has the keys 'internal_votes'
        and 'external_votes' with the number of internal and external votes for that
        candidate.

    Returns
    -------
    list
        A list of the final scores for each candidate, with the external votes
        capped at 80% of the total, and added to the internal votes.
    """
    internal_votes = [item['internal_votes'] for item in context]
    external_votes = [item['external_votes'] for item in context]
    final_scores = []

    for candidate in context:
        if candidate["external_votes"] > 0.8 * (candidate["internal_votes"] + candidate["external_votes"]):
            candidate["external_votes"] = 0.8 * (candidate["internal_votes"] + candidate["external_votes"])
        final_scores.append(candidate["internal_votes"] + candidate["external_votes"])

    return final_scores

def process_votes_external_vote_cap_70(context):
    """
    Process the votes in the context by capping the external votes to 70% of the total.

    Parameters
    ----------
    context : list of dict
        The context list of dictionaries, where each dict has the keys 'internal_votes'
        and 'external_votes' with the number of internal and external votes for that
        candidate.

    Returns
    -------
    list
        A list of the final scores for each candidate, with the external votes
        capped at 70% of the total, and added to the internal votes.
    """
    final_scores = []

    for candidate in context:
        if candidate["external_votes"] > 0.7 * (candidate["internal_votes"] + candidate["external_votes"]):
            candidate["external_votes"] = 0.7 * (candidate["internal_votes"] + candidate["external_votes"])
        final_scores.append(candidate["internal_votes"] + candidate["external_votes"])

    return final_scores

def process_votes_external_vote_cap_60(context):
    """
    Process the votes in the context by capping the external votes to 60% of the total.

    Parameters
    ----------
    context : list of dict
        The context list of dictionaries, where each dict has the keys 'internal_votes'
        and 'external_votes' with the number of internal and external votes for that
        candidate.

    Returns
    -------
    list
        A list of the final scores for each candidate, with the external votes
        capped at 60% of the total, and added to the internal votes.
    """
    internal_votes = [item['internal_votes'] for item in context]
    external_votes = [item['external_votes'] for item in context]
    final_scores = []

    for candidate in context:
        if candidate["external_votes"] > 0.6 * (candidate["internal_votes"] + candidate["external_votes"]):
            candidate["external_votes"] = 0.6 * (candidate["internal_votes"] + candidate["external_votes"])
        final_scores.append(candidate["internal_votes"] + candidate["external_votes"])

    return final_scores

def process_votes_external_vote_cap_50(context):
    """
    Process the votes in the context by capping the external votes to 50% of the total.

    Parameters
    ----------
    context : list of dict
        The context list of dictionaries, where each dict has the keys 'internal_votes'
        and 'external_votes' with the number of internal and external votes for that
        candidate.

    Returns
    -------
    list
        A list of the final scores for each candidate, with the external votes
        capped at 50% of the total, and added to the internal votes.
    """
    internal_votes = [item['internal_votes'] for item in context]
    external_votes = [item['external_votes'] for item in context]
    final_scores = []

    for candidate in context:
        if candidate["external_votes"] > 0.5 * (candidate["internal_votes"] + candidate["external_votes"]):
            candidate["external_votes"] = 0.5 * (candidate["internal_votes"] + candidate["external_votes"])
        final_scores.append(candidate["internal_votes"] + candidate["external_votes"])

    return final_scores

def process_votes_external_vote_cap_40(context):
    """
    Process the votes in the context by capping the external votes to 50% of the total.

    Parameters
    ----------
    context : list of dict
        The context list of dictionaries, where each dict has the keys 'internal_votes'
        and 'external_votes' with the number of internal and external votes for that
        candidate.

    Returns
    -------
    list
        A list of the final scores for each candidate, with the external votes
        capped at 40% of the total, and added to the internal votes.
    """
    internal_votes = [item['internal_votes'] for item in context]
    external_votes = [item['external_votes'] for item in context]
    final_scores = []

    for candidate in context:
        if candidate["external_votes"] > 0.4 * (candidate["internal_votes"] + candidate["external_votes"]):
            candidate["external_votes"] = 0.4 * (candidate["internal_votes"] + candidate["external_votes"])
        final_scores.append(candidate["internal_votes"] + candidate["external_votes"])

    return final_scores

def find_winner(context, voting_rules):
    """
    Process the voting results according to the given voting rules.

    Parameters
    ----------
    context : list of dict
        The context list of dictionaries, where each dict has the keys 'internal_votes'
        and 'external_votes' with the number of internal and external votes for that
        candidate.
    voting_rules : str
        The voting rules to apply to the context.

    Returns
    -------
    list
        A list of the final scores for each candidate, with the internal votes
        doubled and added to the external votes, and the external votes capped at
        60% of the total. The list is sorted in descending order of the final score.

    Raises
    ------
    ValueError
        If the voting rules are invalid.
    """
    results = copy.deepcopy(context)

    if voting_rules == "current":
      ranked_candidates_with_rules = [item['no_rules_candidate_score'] for item in context]

    elif voting_rules == "internal_gets_2_points":
      ranked_candidates_with_rules = process_votes_internal_gets_2_points(context)

    elif voting_rules == "internal_gets_3_points":
      ranked_candidates_with_rules = process_votes_internal_gets_3_points(context)

    elif voting_rules == "external_vote_cap_40":
      ranked_candidates_with_rules = process_votes_external_vote_cap_40(context)

    elif voting_rules == "external_vote_cap_50":
      ranked_candidates_with_rules = process_votes_external_vote_cap_50(context)
    
    elif voting_rules == "external_vote_cap_60":
      ranked_candidates_with_rules = process_votes_external_vote_cap_60(context)

    elif voting_rules == "external_vote_cap_70":
      ranked_candidates_with_rules = process_votes_external_vote_cap_70(context)

    elif voting_rules == "external_vote_cap_80":
      ranked_candidates_with_rules = process_votes_external_vote_cap_80(context)

    else:
      raise ValueError("Invalid voting rules")

    # Reset internal and external votes in context
    for candidate in results:
        candidate["post_rules_candidate_score"] = 0
        candidate["post_rules_candidate_rank"] = 0

    for candidate_id, item in enumerate(results):
        item["post_rules_candidate_score"] = ranked_candidates_with_rules[candidate_id]

    score_sorted = sorted(results, key=lambda x: x["post_rules_candidate_score"], reverse=True)
    for rank, candidate in enumerate(score_sorted, start=1):
        candidate["post_rules_candidate_rank"] = rank

    return results

def calculate_spearman_correlation(results): 
    """
    Calculate the Spearman rank correlation between the pre- and post-rule rankings and the internal ranking.

    Parameters
    ----------
    results : list of dict
        The list of dictionaries, where each dict has the keys 'no_rules_candidate_rank', 'post_rules_candidate_rank', and 'internal_candidate_rank' with the rankings of each candidate in the no-rules, post-rules, and internal rankings, respectively.

    Returns
    -------
    tuple
        A tuple containing the Spearman rank correlation between the no-rules and post-rules rankings, and the Spearman rank correlation between the internal and post-rules rankings, both multiplied by 100.

    """
    # Extract the rankings from the results
    no_rules_ranks = [candidate['no_rules_candidate_rank'] for candidate in results]
    post_rules_ranks = [candidate['post_rules_candidate_rank'] for candidate in results]

    # Calculate Spearman's rank correlation
    representation_correlation, representation_p_value = spearmanr(no_rules_ranks, post_rules_ranks)

    internal_rank = [candidate['internal_candidate_rank'] for candidate in results]

    internal_approval_correlation, internal_approval_p_value = spearmanr(internal_rank, post_rules_ranks)

    return max(0, representation_correlation*100), max(0, internal_approval_correlation*100)

def process_voting_results(context, voting_rules):
    """
    Process the voting results according to the given voting rules.

    Parameters
    ----------
    context : list of dict
        The context list of dictionaries, where each dict has the keys 'internal_votes'
        and 'external_votes' with the number of internal and external votes for that
        candidate.
    voting_rules : str
        The voting rules to apply to the context.
    internal_voting_intention : list
        The internal voting intention list of probabilities.
    external_voting_intention : list
        The external voting intention list of probabilities.
    no_of_candidates : int
        The number of candidates.

    Returns
    -------
    tuple
        A tuple containing the results of the election, the representation index (Spearman rank
        correlation between the pre- and post-rules rankings), and the internal approval index
        (Spearman rank correlation between the internal and post-rules rankings), all multiplied by 100.

    """
    results = find_winner(context, voting_rules)
    representation_index, internal_approval_index = calculate_spearman_correlation(results)

    return results, representation_index, internal_approval_index