# Complete offline genome examples

Both reference DNA sequences and their protein queries are included in the installed package. No download, API key, or network access is needed.

- `mouse-mt`: the complete mouse mitochondrial chromosome, **NC_005089.1**, 16,299 bp; query **COX1 / NP_904330.1**, 514 amino acids. Uses NCBI translation table **2** (vertebrate mitochondrial).
- `ecoli`: the complete *Escherichia coli* K-12 MG1655 chromosome, **NC_000913.3**, 4,641,652 bp; query **lacZ / NP_414878.1**, 1,024 amino acids. Uses translation table **11** (bacterial).

The DNA sequences total **4,657,951 bp**. FASTA data and queries occupy about **4.72 MB** uncompressed. They are complete reference sequences, not gene-region extracts. This example demonstrates intronless mitochondrial gene alignment, not nuclear splicing. Neither genome includes a gene crossing the linearized origin in its selected query locus.

## Run after installation

```bash
protlocus-example --list
protlocus-example mouse-mt
protlocus-example ecoli
```

The default backend is CPU. Each run writes into `protlocus-example-NAME/` in the current directory. Use a fresh `--outdir` to repeat a run. The installed reference data are never modified.

```bash
protlocus-example ecoli --outdir ecoli-results --threads 4
# Requires the corresponding GPU package, device and driver:
protlocus-example ecoli --backend cuda --outdir ecoli-cuda
protlocus-example mouse-mt --backend opencl --outdir mouse-opencl
```

GPU requests fail if device initialization cannot be confirmed. These small examples demonstrate functionality; they are not speed benchmarks and may run faster on CPU.

For a source checkout, build with `make` and run `python3 tools/protlocus-example mouse-mt`. Native binary bundles provide `bin/protlocus-example`; add their `bin/` directory to PATH or invoke it directly. Conda installs the command in `bin/` and the data in `share/protlocus/examples/`; the CPU package supplies the data for all three backends.

## Results and expected loci

- `result.gff3`: alignment-derived gene features.
- `alignments.paf`: protein-to-genome alignments.
- `run.log`: the command and diagnostic output.
- `report.json`: validation of the known gene locus, strand, and at least 95% query coverage.

COX1 lies at **NC_005089.1:5328–6872 (+)** and lacZ at **NC_000913.3:363231–366305 (−)**, using 1-based inclusive annotation coordinates. Alignments normally omit the terminal stop codon; PAF uses 0-based, half-open coordinates. These are reference self-alignment demonstrations, not independent accuracy benchmarks or whole-genome gene annotations.

## Data provenance

Reference DNA and annotated CDS protein translations were retrieved from NCBI on 2026-11-03. Each dataset's `metadata.json` records accession versions, record/retrieval URLs, translation table, annotated locus, full sequence length, and SHA256 checksums. FASTA sequences were reformatted to 80 columns without truncation or sequence edits. The E. coli FASTA was also checked against the independently retrieved GenBank sequence.

- Mouse record: https://www.ncbi.nlm.nih.gov/nuccore/NC_005089.1
- E. coli record: https://www.ncbi.nlm.nih.gov/nuccore/NC_000913.3

Data credit belongs to the original sequence submitters and reference curators identified in these records. The software's MIT license applies to the project code; these reference sequences retain their original data provenance. See NCBI's usage policies: https://www.ncbi.nlm.nih.gov/home/about/policies/
