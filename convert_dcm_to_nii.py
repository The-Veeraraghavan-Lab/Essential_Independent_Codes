import SimpleITK as sitk
import os
import pydicom
import pandas as pd
import re
from datetime import datetime


def find_dcm_folders(root_dir):
    dcm_folders = set()
    for dirpath, _, filenames in os.walk(root_dir):
        if any(file.endswith(".dcm") for file in filenames):
            dcm_folders.add(dirpath)
    return sorted(dcm_folders)


def get_series_metadata(dicom_folder):
    dcm_files = [f for f in os.listdir(dicom_folder) if f.endswith(".dcm")]
    if not dcm_files:
        return None
    sample_dcm = pydicom.dcmread(
        os.path.join(dicom_folder, dcm_files[0]), stop_before_pixels=True
    )
    return {
        "Folder": dicom_folder,
        "Modality": getattr(sample_dcm, "Modality", ""),
        "Series Description": getattr(sample_dcm, "SeriesDescription", ""),
        "Series Instance UID": getattr(sample_dcm, "SeriesInstanceUID", ""),
        "Number of Images": len(dcm_files),
    }


def choose_main_ct(study_df):
    df = study_df.copy()
    df = df[df["Modality"].str.upper() == "CT"].copy()
    if df.empty:
        return None
    desc = df["Series Description"].fillna("").str.lower()
    reject_pattern = r"scout|localizer|topo|topogram|cor\b|sag\b|mpr"
    df = df[~desc.str.contains(reject_pattern, regex=True)].copy()
    if df.empty:
        return None
    desc = df["Series Description"].fillna("").str.lower()
    score = pd.Series(0, index=df.index, dtype=float)
    score += desc.str.contains(r"routine", regex=True).astype(int) * 5
    score += desc.str.contains(r"chest", regex=True).astype(int) * 4
    score += desc.str.contains(r"non[- ]?con|without contrast", regex=True).astype(int) * 3
    score += desc.str.contains(r"axial|ax", regex=True).astype(int) * 2
    score += desc.str.contains(r"standard", regex=True).astype(int) * 2
    score -= desc.str.contains(r"bone", regex=True).astype(int) * 6
    score -= desc.str.contains(r"alg|algorithm|kernel", regex=True).astype(int) * 3
    score -= desc.str.contains(r"thin|0\.625|1\.0 ?mm|1mm|1\.25", regex=True).astype(int) * 4
    score -= desc.str.contains(r"lung", regex=True).astype(int) * 2
    score -= desc.str.contains(r"recon|reformat", regex=True).astype(int) * 3
    nimg = pd.to_numeric(df["Number of Images"], errors="coerce").fillna(0)
    score += (nimg.between(80, 400)).astype(int) * 2
    score += (nimg.between(150, 350)).astype(int) * 2
    df = df.assign(score=score)
    df = df.sort_values(["score", "Number of Images"], ascending=[False, False])
    return df.iloc[0]


def dicom_to_nifti(dicom_dir, output_nifti_path):
    reader = sitk.ImageSeriesReader()

    series_uids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_uids:
        raise ValueError(f"No DICOM series found in {dicom_dir}")

    # For each series, also filter to a consistent matrix size within it
    def get_consistent_files(dicom_dir, uid):
        files = list(reader.GetGDCMSeriesFileNames(dicom_dir, uid))
        if not files:
            return files

        # Read the matrix size of each slice and keep only the dominant shape
        from collections import Counter
        shape_map = {}
        for f in files:
            try:
                dcm = pydicom.dcmread(f, stop_before_pixels=True)
                shape = (int(dcm.Rows), int(dcm.Columns))
                shape_map.setdefault(shape, []).append(f)
            except Exception:
                continue

        if not shape_map:
            return files

        # Pick the shape with the most slices
        dominant_shape = max(shape_map, key=lambda s: len(shape_map[s]))
        kept = shape_map[dominant_shape]

        if len(kept) < len(files):
            print(f"  [mixed geometry] {uid}: kept {len(kept)}/{len(files)} slices "
                  f"with shape {dominant_shape}, "
                  f"dropped {len(files) - len(kept)} outlier slice(s)")

        return kept

    # Pick the series UID whose dominant-shape slice count is largest
    uid_files = {uid: get_consistent_files(dicom_dir, uid) for uid in series_uids}
    chosen_uid = max(uid_files, key=lambda uid: len(uid_files[uid]))

    if len(series_uids) > 1:
        print(f"  [multi-series folder] {len(series_uids)} series, "
              f"picked UID {chosen_uid} ({len(uid_files[chosen_uid])} usable slices)")

    consistent_files = uid_files[chosen_uid]
    if not consistent_files:
        raise ValueError(f"No consistent slices found in {dicom_dir}")

    reader.SetFileNames(consistent_files)
    image = reader.Execute()
    sitk.WriteImage(image, output_nifti_path)


def group_folders_by_patient(dcm_folders):
    groups = {}
    for folder in dcm_folders:
        parts = folder.split("/")
        patient_id = parts[1] if len(parts) > 1 else folder
        groups.setdefault(patient_id, []).append(folder)
    return groups


# ── Main ────────────────────────────────────────────────────────────────────

INPUT_DIR  = "MIDRC-RICORD-1a"
OUTPUT_DIR = "COVID-19-a"
LOG_PATH   = "conversion_log.txt"

# ── Overrides ────────────────────────────────────────────────────────────────
# If non-empty, ONLY these patients are processed and the folder is used directly
# (no scoring). Leave empty — {} — to run the full dataset with auto-selection.
#
# Format:  "patient-id": "path/to/dicom/folder"
#
OVERRIDES = {
    "MIDRC-RICORD-1A-419639-001476":
        "MIDRC-RICORD-1a/MIDRC-RICORD-1A-419639-001476/07-19-2009-NA-CT CHEST WITH CONTRAST-53644/3.000000-CTA  SP ON ASCENDING AORTA-94442",
    "MIDRC-RICORD-1A-660042-000049":
        "MIDRC-RICORD-1a/MIDRC-RICORD-1A-660042-000049/12-19-2000-NA-NA-11239/3.000000-NA-11240",
    "MIDRC-RICORD-1A-419639-001336":
        "MIDRC-RICORD-1a/MIDRC-RICORD-1A-419639-001336/10-27-2008-NA-CT ANGIOGRAM CHEST-85568/3.000000-1.25 60 KEV-06399",
    "MIDRC-RICORD-1A-419639-001533":
        "MIDRC-RICORD-1a/MIDRC-RICORD-1A-419639-001533/09-23-2001-NA-CT CHEST PULMONARY EMBOLISM CTPE-46200/3.000000-1.25 60 KEV-69058"
}
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

if OVERRIDES:
    # Only process the explicitly specified patients/folders
    patient_groups = {pid: [folder] for pid, folder in OVERRIDES.items()}
    log_mode = "a"   # append — don't wipe the existing log
    print(f"Override mode: processing {len(OVERRIDES)} patient(s) only.")
else:
    all_dcm_folders = find_dcm_folders(INPUT_DIR)
    patient_groups  = group_folders_by_patient(all_dcm_folders)
    log_mode = "w"   # fresh log for a full run

with open(LOG_PATH, log_mode) as log:
    if OVERRIDES:
        log.write(f"\n{'=' * 80}\n")
        log.write(f"OVERRIDE run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                  f"— {len(OVERRIDES)} patient(s)\n")
        log.write(f"{'=' * 80}\n\n")
    else:
        log.write(f"Conversion run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Total patients: {len(patient_groups)}\n")
        log.write("=" * 80 + "\n\n")

    for patient_id, folders in patient_groups.items():
        output_path = os.path.join(OUTPUT_DIR, f"{patient_id}.nii.gz")

        if OVERRIDES:
            # Folder was specified explicitly — use it directly, skip scoring
            chosen_folder = folders[0]
            log.write(f"[OVERRIDE] {patient_id}\n")
            log.write(f"           folder: {chosen_folder}\n")
        elif len(folders) == 1:
            chosen_folder = folders[0]
            log.write(f"[OK]   {patient_id}\n")
            log.write(f"       single series → {chosen_folder}\n")
        else:
            records = [get_series_metadata(f) for f in folders]
            records = [r for r in records if r is not None]

            if not records:
                msg = f"[SKIP] {patient_id} — no readable DICOM files\n\n"
                print(msg.strip()); log.write(msg)
                continue

            study_df = pd.DataFrame(records)
            chosen   = choose_main_ct(study_df)

            if chosen is None:
                msg = f"[SKIP] {patient_id} — no valid CT series after filtering\n"
                print(msg.strip()); log.write(msg)
                for _, row in study_df.iterrows():
                    log.write(
                        f"       rejected: '{row['Series Description']}' "
                        f"mod={row['Modality']}  n={row['Number of Images']}\n"
                    )
                log.write("\n")
                continue

            chosen_folder = chosen["Folder"]
            log.write(
                f"[OK]   {patient_id}\n"
                f"       chose:  '{chosen['Series Description']}'  "
                f"score={chosen['score']:.0f}  n={chosen['Number of Images']}\n"
                f"       folder: {chosen_folder}\n"
            )
            others = study_df[study_df["Folder"] != chosen_folder]
            for _, row in others.iterrows():
                log.write(
                    f"       other:  '{row['Series Description']}'  "
                    f"mod={row['Modality']}  n={row['Number of Images']}  "
                    f"← {row['Folder']}\n"
                )

        try:
            dicom_to_nifti(chosen_folder, output_path)
            log.write(f"       saved:  {output_path}\n\n")
            print(f"[OK] {patient_id} → {output_path}")
        except Exception as e:
            msg = f"[ERR]  {patient_id} — conversion failed: {e}\n\n"
            print(msg.strip()); log.write(msg)
