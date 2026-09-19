from pathlib import Path

from src.submission.writers import ranked_ids_to_positions, validate_prediction_file


def test_ranked_ids_to_positions():
    # Candidates N1, N2, N3 ranked as N2 (best, rank 1), N3 (2nd, rank 2), N1 (3rd, rank 3)
    # Output must be ranks for [N1, N2, N3] -> [3, 1, 2]
    assert ranked_ids_to_positions(["N1", "N2", "N3"], ["N2", "N3", "N1"]) == [3, 1, 2]


def test_validate_prediction_file(tmp_path: Path):
    path = tmp_path / "prediction.txt"
    path.write_text("1 [2,1,3]\n2 [1,2]\n")
    assert validate_prediction_file(path, expected_rows=2) == 2
