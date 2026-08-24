#!/usr/bin/env bash
set -euo pipefail

PROJECT="/Users/bhaktidoshi/Downloads/Thesis Data"
STAGE4="$PROJECT/thesis_preprocessing_workflow/stage4_multiomics_integration"
SAMPLES="$STAGE4/manifests/ihmp_metaphlan_samples_to_run.txt"
RAW_DIR="$STAGE4/data/external_microbiome/ihmp/raw_mgx"
PROFILE_DIR="$STAGE4/data/external_microbiome/ihmp/metaphlan_profiles"
mkdir -p "$RAW_DIR" "$PROFILE_DIR"

if ! command -v metaphlan >/dev/null 2>&1; then
  echo "MetaPhlAn is not available on PATH. Install/configure MetaPhlAn before running." >&2
  exit 1
fi

while IFS= read -r SAMPLE; do
  [ -z "$SAMPLE" ] && continue
  echo "Prepare raw/per-sample processed input for $SAMPLE manually from the authoritative HMP2/IBDMDB route, then run MetaPhlAn here."
  echo "Expected raw candidates documented in stage4a_ihmp_raw_reconstruction_manifest.csv"
  # Example placeholder once FASTQ files are available:
  # metaphlan "$RAW_DIR/${SAMPLE}_R1.fastq.gz,$RAW_DIR/${SAMPLE}_R2.fastq.gz" --input_type fastq -o "$PROFILE_DIR/${SAMPLE}_metaphlan_profile.tsv" --nproc 4
done < "$SAMPLES"
