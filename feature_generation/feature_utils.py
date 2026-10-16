def parse_fasta_file(file_path: str) -> dict:
    protein_dict: dict = {}
    current_id: str | None = None
    sequence: str = ""
    label: str = ""

    with open(file_path, "r") as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith(">"):
                if current_id is not None:
                    protein_dict[current_id] = [sequence, label]
                current_id = line[1:]
                sequence = ""
                label = ""
            elif current_id is not None:
                if not sequence:
                    sequence = line
                else:
                    label = line

    if current_id is not None:
        protein_dict[current_id] = [sequence, label]

    return protein_dict
