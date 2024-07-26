"""
Sara Baldinelli
Algorithms for bioinformatics exam
31/05/2024
"""

import numpy as np
import argparse

# Function to create a substitution matrix with match and mismatch scores
def create_substitution_matrix(match_score, mismatch_score):
    """
    Args:
        match_score (int): score for matching characters
        mismatch_score (int): score for mismatching characters
    Returns:
        dict: substitution matrix with match and mismatch scores
    """
    return {'match': match_score, 'mismatch': mismatch_score}

# Function to initialize scoring, gapA, and gapB matrices
def initialize_matrices(n, m):
    """
    Args:
        n (int):length of the first sequence
        m (int): length of the second sequence
    Returns:
        tuple: initialized scoring, gapA, and gapB matrices
    """
    scoring_matrix = np.zeros((n + 1, m + 1), dtype=int)
    gapA_matrix = np.zeros((n + 1, m + 1), dtype=int)
    gapB_matrix = np.zeros((n + 1, m + 1), dtype=int)
    return scoring_matrix, gapA_matrix, gapB_matrix

# Function to fill the scoring, gapA, gapB, and traceback matrix
def fill_scoring_matrix(seq1, seq2, sub_matrix, gap_open_penalty, gap_extend_penalty):
    """
    Args:
        seq1 (str): First sequence
        seq2 (str): Second sequence
        sub_matrix (dict): Substitution matrix containing match and mismatch scores
        gap_open_penalty (int): Penalty for opening a gap
        gap_extend_penalty (int): Penalty for extending a gap
    Returns:
        tuple: Filled scoring, gapA, gapB, and traceback matrices, and max score positions
    """
    n = len(seq1)
    m = len(seq2)
    
    #Initialize matrices for scoring and gap penalties
    scoring_matrix, gapA_matrix, gapB_matrix = initialize_matrices(n, m)
    traceback_matrix = np.zeros((n + 1, m + 1), dtype=int)  # Matrix to store traceback information
    
    max_score = 0  
    max_positions = []  

    # loop through each cell in the matrices
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            #calculate match/mismatch score based on the previous diagonal cell
            match = scoring_matrix[i-1][j-1] + (sub_matrix['match'] if seq1[i-1] == seq2[j-1] else sub_matrix['mismatch'])
            
            #calculate deletion score (gap in seq2)
            delete = max(scoring_matrix[i-1][j] + gap_open_penalty, gapA_matrix[i-1][j] + gap_extend_penalty)
            
            #calculate insertion score (gap in seq1)
            insert = max(scoring_matrix[i][j-1] + gap_open_penalty, gapB_matrix[i][j-1] + gap_extend_penalty)
            
            # Fill the scoring matrix with the highest score (0 for local alignment)
            scoring_matrix[i][j] = max(0, match, delete, insert)
            
            # Update gap matrices
            gapA_matrix[i][j] = max(scoring_matrix[i-1][j] + gap_open_penalty, gapA_matrix[i-1][j] + gap_extend_penalty)
            gapB_matrix[i][j] = max(scoring_matrix[i][j-1] + gap_open_penalty, gapB_matrix[i][j-1] + gap_extend_penalty)
            
            # Update the traceback matrix with the direction of the highest score
            if scoring_matrix[i][j] == match:
                traceback_matrix[i][j] = 1  #1 indicates a match/mismatch (diagonal move)
            elif scoring_matrix[i][j] == delete:
                traceback_matrix[i][j] = 2  #2 indicates a deletion (upward move)
            elif scoring_matrix[i][j] == insert:
                traceback_matrix[i][j] = 3  #3 indicates an insertion (leftward move)

            # Track the maximum score and its positions
            if scoring_matrix[i][j] > max_score:
                max_score = scoring_matrix[i][j]
                max_positions = [(i, j)]  #reset positions with the new max score
            elif scoring_matrix[i][j] == max_score:
                max_positions.append((i, j))  #apend the position of the equal max score
    
    # Return the filled matrices and positions of the maximum scores
    return scoring_matrix, traceback_matrix, max_positions

# Function to perform traceback to generate alignments
def traceback(scoring_matrix, traceback_matrix, start_pos, seq1, seq2):
    """
    Args:
        scoring_matrix (ndarray): Filled scoring matrix
        traceback_matrix (ndarray): Traceback matrix
        start_pos (tuple): Starting position for traceback
        seq1 (str): First sequence
        seq2 (str): Second sequence
    Returns:
        tuple: Aligned sequences, alignment score
    """
    aligned_seq1 = []  # List to store the aligned sequence 1
    aligned_seq2 = []  # List to store the aligned sequence 2
    i, j = start_pos  # Initialize traceback starting position
    alignment_score = scoring_matrix[i][j]  # Alignment score is the score at the start position

    # Perform traceback until a cell with score 0 is reached
    while scoring_matrix[i][j] > 0:
        if traceback_matrix[i][j] == 1:
            # Match/mismatch case: move diagonally
            aligned_seq1.append(seq1[i-1])
            aligned_seq2.append(seq2[j-1])
            i -= 1
            j -= 1
        elif traceback_matrix[i][j] == 2:
            # Deletion case: move up (gap in seq2)
            aligned_seq1.append(seq1[i-1])
            aligned_seq2.append('-')
            i -= 1
        elif traceback_matrix[i][j] == 3:
            # Insertion case: move left (gap in seq1)
            aligned_seq1.append('-')
            aligned_seq2.append(seq2[j-1])
            j -= 1

    # Reverse the lists to obtain the correct alignment order
    aligned_seq1.reverse()
    aligned_seq2.reverse()

    # Return the aligned sequences as strings and the aligement score
    return ''.join(aligned_seq1), ''.join(aligned_seq2), alignment_score


# Function to find all alignments that meet the minimum score percentage, if chosen
def find_all_alignments(seq1, seq2, sub_matrix, gap_open_penalty, gap_extend_penalty, min_score_percentage):
    """
    Args:
        seq1 (str): First sequence
        seq2 (str): Second sequence
        sub_matrix (dict): Substitution matrix containing match and mismatch scores
        gap_open_penalty (int): Penalty for opening a gap
        gap_extend_penalty (int): Penalty for extending a gap
        min_score_percentage (float): Minimum score percentage for alignments to be considered
    Returns:
        list: List of valid alignments (tuples of aligned sequences, score, length, number of gaps, score percentage)
    """
    # Fill the scoring and traceback matrices
    scoring_matrix, traceback_matrix, max_positions = fill_scoring_matrix(seq1, seq2, sub_matrix, gap_open_penalty, gap_extend_penalty)
    alignments = []  #List to store all valid alignments

    visited = set()  # Set to keep track of visited positions in the scoring matrix
    max_possible_score = max(len(seq1),len(seq2)) * sub_matrix['match']  #maximum possible score
    min_score = (min_score_percentage/100) * max_possible_score  # minimum score treshold based on the selected percentage

    def get_next_highest_position(scoring_matrix, visited):
        """
        Find the next highest scoring position in the scoring matrix that hasn't been visited
        Args:
            scoring_matrix (ndarray)
            visited (set): Set of visited positions
        Returns:
            tuple: Coordinates of the next highest scoring position
        """
        max_score = 0  
        max_pos = None  
        for i in range(1, scoring_matrix.shape[0]):
            for j in range(1, scoring_matrix.shape[1]):
                # Check if the current position has not been visited and has a higher score than max_score
                if (i, j) not in visited and scoring_matrix[i][j] > max_score:
                    max_score = scoring_matrix[i][j]  # update max_score to the current cell's score
                    max_pos = (i, j)  # update max_pos to the current cell's coordinates
        return max_pos  # Return the coordinates of the cell with the highest score found


    while True:
        pos = get_next_highest_position(scoring_matrix, visited)  # find the next highest scoring position
        if pos is None or scoring_matrix[pos] == 0:  #break if no valid position is found
            break
        visited.add(pos)  # mark position as visited
        aligned_seq1, aligned_seq2, score = traceback(scoring_matrix, traceback_matrix, pos, seq1, seq2)  # Perform traceback
        alignment_length = len(aligned_seq1)  # lenght of the alignment
        num_gaps = aligned_seq1.count('-') + aligned_seq2.count('-')  #Number of gaps in the alignment
        alignment_score_percentage = (score/max_possible_score) * 100  #alignment score percentage

        if score >= min_score:  # Check if the score meets the minimum threshold
            alignments.append((aligned_seq1, aligned_seq2, score, alignment_length, num_gaps, alignment_score_percentage))

    # Sort alignments by score and then by length in descending order
    alignments.sort(key=lambda x: (x[2],x[3]),reverse=True)
    return alignments


# Main function to run the Smith-Waterman algorithm
def main(seq1, seq2, match_score, mismatch_score, gap_open_penalty, gap_extend_penalty, min_score_percentage):
    """Main function to run the Smith-Waterman algorithm with affine gap penalty
    Args:
        seq1 (str): the first sequence
        seq2 (str): the second sequence
        match_score (int): score for matching characters
        mismatch_score (int): score for mismatching characters
        gap_open_penalty (int): penalty for opening a gap
        gap_extend_penalty (int): penalty for extending a gap
        min_score_percentage (float): minimum alignment score as a percentage
    """
    sub_matrix = create_substitution_matrix(match_score, mismatch_score)
    alignments = find_all_alignments(seq1, seq2, sub_matrix, gap_open_penalty, gap_extend_penalty, min_score_percentage)

    if not alignments:
        print("No alignments found with the given criteria.")
    else:
        print(f"\nInput Sequences:\nseq1: {seq1}\nseq2: {seq2}")
        print("----------------------------------------------------------------------")
        for i, (aligned_seq1, aligned_seq2, score, alignment_length, num_gaps, alignment_score_percentage) in enumerate(alignments):
            symbols = ''.join(['*' if a == b else '|' if a != '-' and b != '-' else ' ' for a, b in zip(aligned_seq1, aligned_seq2)])
            num_matches = symbols.count('*')
            num_mismatches = symbols.count('|')
            print(f"Alignment {i + 1}:")
            print(f"\t{aligned_seq1}")
            print(f"\t{symbols}")
            print(f"\t{aligned_seq2}")
            print(f"The score of the alignment: {score}\nAlignment length: {alignment_length}\nNumber of matches: {num_matches}\nNumber of mismatches: {num_mismatches}\nNumber of gaps: {num_gaps}\nAlignment score percentage: {alignment_score_percentage:.2f}%")
            print(f"----------------------------------------------------------------------")

# Entry point for the script with argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Implementation of the Smith-Waterman local alignment with affine gap penalty")
    parser.add_argument("seq1", type=str, help="Insert the first sequence, the reference to which align the second sequence.")
    parser.add_argument("seq2", type=str, help="Insert the second sequence, sequence to align to the first one.")
    parser.add_argument("-m","--match", type=int, default=1, help="Change this parameter in order to change the match score [default 1].")
    parser.add_argument("-p","--mismatch", type=int, default=-1, help="Change this parameter in order to change the mismatch score [default -1].")
    parser.add_argument("-o","--gap_open", type=int, default=-3, help="Change this parameter in order to change the gap opening penalty [default -3].")
    parser.add_argument("-e","--gap_extend", type=int, default=-1, help="Change this parameter in order to change the gap enlargment penalty [default -1].")
    parser.add_argument("-s","--min_score_percentage", type=float, default=0, help="Choose the minimum percentage for the alignment score, in order to filter out the lowest scoring alignments [default 0].")

    args = parser.parse_args()
    main(args.seq1, args.seq2, args.match, args.mismatch, args.gap_open, args.gap_extend, args.min_score_percentage)


"""
Line of code from the terminal:
python3 SW_Baldinelli.py [-h] [-m MATCH] [-p MISMATCH] [-o GAP_OPEN] [-e GAP_EXTEND] [-s MIN_SCORE_PERCENTAGE] seq1 seq2

Options:
First sequence [required] the reference to which align the second sequence.
Second sequence [required] the sequence to align to the first one.
-h, --help [optional] print the help message and exit.
-m, --match [optional] change the score for a base match [default 1]
-p, --mismatch [optional] change the penalty for a base mismatch [default -1]
-o, --gap_open [optional] change the penalty for a gap opening [default -3]
-e, --gap_extend [optional] change the penalty for a gap extend [default -1]
-s, --min_score_percentage [optional] the minumum score for a solution to be included [default 0]
"""