#!/usr/bin/env python3
"""Collect the fine-tuning corpus: Pakistani identities disjoint from the eval set.

Why this exists
---------------
The brief requires teams to collect their own training data and fine-tune. The
evaluation identities in `data/pak_fig` must never appear here, so the two sets
are checked for overlap before anything is written.

What it collects
----------------
`data/raw/train_faces/<identity>/` — raw scraped images, one folder per person,
with a metadata row recording gender and whether the person is habitually veiled
(hijab/dupatta) in public appearances. That flag matters for the fairness
analysis: the corpus is deliberately weighted towards women who wear a head
covering, because the evaluation benchmark (§7.4) is where the pipeline is
weakest.

What it cannot collect
----------------------
Real niqab photos with identity labels do not exist publicly. The 532 real niqab
images we do have are used for evaluation, never for supervised training. Niqab
robustness is therefore trained synthetically: the training job paints occluders
over the lower face of labelled identities. This script only gathers the
labelled part.

Usage
-----
    python src/collect_train_faces.py --per-id 45
    python src/collect_train_faces.py --per-id 45 --only ayesha_omar,atif_aslam
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from icrawler.builtin import BingImageCrawler

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "train_faces"

# (slug, search query, gender, veiled-in-public flag)
# `veiled` = wears hijab/dupatta in a large share of public photos. It is a
# coarse curation label used to weight the corpus, not a verified attribute.
PEOPLE = [
    # ---- women: entertainment, many routinely in dupatta/hijab ----
    ("mahira_khan", "Mahira Khan Pakistani actress", "f", 1),
    ("sajal_aly", "Sajal Aly Pakistani actress", "f", 1),
    ("hania_aamir", "Hania Aamir Pakistani actress", "f", 0),
    ("maya_ali", "Maya Ali Pakistani actress", "f", 1),
    ("saba_qamar", "Saba Qamar Pakistani actress", "f", 0),
    ("mehwish_hayat", "Mehwish Hayat Pakistani actress", "f", 0),
    ("ayesha_omar", "Ayesha Omar Pakistani actress", "f", 0),
    ("sanam_saeed", "Sanam Saeed Pakistani actress", "f", 1),
    ("aamina_sheikh", "Aamina Sheikh Pakistani actress", "f", 1),
    ("iqra_aziz", "Iqra Aziz Pakistani actress", "f", 1),
    ("yumna_zaidi", "Yumna Zaidi Pakistani actress", "f", 1),
    ("kubra_khan", "Kubra Khan Pakistani actress", "f", 1),
    ("sarah_khan", "Sarah Khan Pakistani actress", "f", 1),
    ("aiman_khan", "Aiman Khan Pakistani actress", "f", 1),
    ("minal_khan", "Minal Khan Pakistani actress", "f", 1),
    ("neelam_muneer", "Neelam Muneer Pakistani actress", "f", 1),
    ("ushna_shah", "Ushna Shah Pakistani actress", "f", 0),
    ("sarwat_gilani", "Sarwat Gilani Pakistani actress", "f", 1),
    ("nadia_jamil", "Nadia Jamil Pakistani actress", "f", 1),
    ("sanam_baloch", "Sanam Baloch Pakistani actress", "f", 1),
    ("hareem_farooq", "Hareem Farooq Pakistani actress", "f", 1),
    ("zara_noor_abbas", "Zara Noor Abbas Pakistani actress", "f", 1),
    ("alizeh_shah", "Alizeh Shah Pakistani actress", "f", 0),
    ("nimra_khan", "Nimra Khan Pakistani actress", "f", 0),
    ("areeba_habib", "Areeba Habib Pakistani actress", "f", 0),
    ("ghana_ali", "Ghana Ali Pakistani actress", "f", 1),
    ("meera", "Meera Pakistani actress", "f", 0),
    ("reema_khan", "Reema Khan Pakistani actress", "f", 0),
    ("atiqa_odho", "Atiqa Odho Pakistani actress", "f", 0),
    ("bushra_ansari", "Bushra Ansari Pakistani actress", "f", 0),
    ("samina_peerzada", "Samina Peerzada Pakistani actress", "f", 0),
    ("momal_sheikh", "Momal Sheikh Pakistani actress", "f", 1),
    ("urwa_hocane", "Urwa Hocane Pakistani actress", "f", 0),
    ("mawra_hocane", "Mawra Hocane Pakistani actress", "f", 0),
    ("zhalay_sarhadi", "Zhalay Sarhadi Pakistani actress", "f", 0),
    # ---- women: music ----
    ("abida_parveen", "Abida Parveen Pakistani singer", "f", 1),
    ("hadiqa_kiani", "Hadiqa Kiani Pakistani singer", "f", 0),
    ("momina_mustehsan", "Momina Mustehsan Pakistani singer", "f", 1),
    ("aroj_aftab", "Arooj Aftab Pakistani singer", "f", 0),
    ("sanam_marvi", "Sanam Marvi Pakistani singer", "f", 1),
    ("farida_khanum", "Farida Khanum Pakistani singer", "f", 0),
    ("nazia_hassan", "Nazia Hassan Pakistani singer", "f", 0),
    ("reshma", "Reshma Pakistani folk singer", "f", 0),
    # ---- women: public life ----
    ("malala_yousafzai", "Malala Yousafzai", "f", 1),
    ("asma_jahangir", "Asma Jahangir Pakistani lawyer", "f", 0),
    ("ruth_pfau", "Ruth Pfau Pakistan", "f", 0),
    ("sherry_rehman", "Sherry Rehman Pakistani politician", "f", 0),
    ("zartaj_gul", "Zartaj Gul Pakistani politician", "f", 1),
    ("maleeka_bokhari", "Maleeka Bokhari Pakistani politician", "f", 0),
    ("sania_nishtar", "Sania Nishtar Pakistani", "f", 0),
    ("marvi_memon", "Marvi Memon Pakistani politician", "f", 0),
    ("sharmila_faruqui", "Sharmila Faruqui Pakistani", "f", 0),
    ("nafisa_shah", "Nafisa Shah Pakistani politician", "f", 0),
    ("fauzia_kasuri", "Fauzia Kasuri Pakistani politician", "f", 0),
    # ---- men: cricket ----
    ("wasim_akram", "Wasim Akram cricketer", "m", 0),
    ("waqar_younis", "Waqar Younis cricketer", "m", 0),
    ("shoaib_akhtar", "Shoaib Akhtar cricketer", "m", 0),
    ("shahid_afridi", "Shahid Afridi cricketer", "m", 0),
    ("babar_azam", "Babar Azam cricketer", "m", 0),
    ("mohammad_rizwan", "Mohammad Rizwan cricketer", "m", 0),
    ("shaheen_afridi", "Shaheen Afridi cricketer", "m", 0),
    ("misbah_ul_haq", "Misbah-ul-Haq cricketer", "m", 0),
    ("younis_khan", "Younis Khan cricketer", "m", 0),
    ("inzamam_ul_haq", "Inzamam-ul-Haq cricketer", "m", 0),
    ("javed_miandad", "Javed Miandad cricketer", "m", 0),
    ("sarfaraz_ahmed", "Sarfaraz Ahmed cricketer", "m", 0),
    ("mohammad_amir", "Mohammad Amir cricketer", "m", 0),
    ("ramiz_raja", "Ramiz Raja cricketer", "m", 0),
    ("saeed_anwar", "Saeed Anwar cricketer", "m", 0),
    # ---- men: screen and music ----
    ("fawad_khan", "Fawad Khan Pakistani actor", "m", 0),
    ("humayun_saeed", "Humayun Saeed Pakistani actor", "m", 0),
    ("shaan_shahid", "Shaan Shahid Pakistani actor", "m", 0),
    ("hamza_ali_abbasi", "Hamza Ali Abbasi actor", "m", 0),
    ("osman_khalid_butt", "Osman Khalid Butt actor", "m", 0),
    ("bilal_abbas", "Bilal Abbas Khan actor", "m", 0),
    ("imran_abbas", "Imran Abbas actor", "m", 0),
    ("feroze_khan", "Feroze Khan actor", "m", 0),
    ("mikaal_zulfiqar", "Mikaal Zulfiqar actor", "m", 0),
    ("ahsan_khan", "Ahsan Khan actor", "m", 0),
    ("danish_taimoor", "Danish Taimoor actor", "m", 0),
    ("noman_ijaz", "Noman Ijaz actor", "m", 0),
    ("adnan_siddiqui", "Adnan Siddiqui actor", "m", 0),
    ("yasir_hussain", "Yasir Hussain actor", "m", 0),
    ("atif_aslam", "Atif Aslam singer", "m", 0),
    ("ali_zafar", "Ali Zafar singer", "m", 0),
    ("rahat_fateh_ali_khan", "Rahat Fateh Ali Khan singer", "m", 0),
    ("nusrat_fateh_ali_khan", "Nusrat Fateh Ali Khan", "m", 0),
    ("ghulam_ali", "Ghulam Ali ghazal singer", "m", 0),
    ("ali_sethi", "Ali Sethi singer", "m", 0),
    ("asim_azhar", "Asim Azhar singer", "m", 0),
    # ---- men: public life ----
    ("abdul_sattar_edhi", "Abdul Sattar Edhi", "m", 0),
    ("aq_khan", "Abdul Qadeer Khan scientist", "m", 0),
    ("anwar_maqsood", "Anwar Maqsood", "m", 0),
    ("moin_akhtar", "Moin Akhtar", "m", 0),
    ("umer_sharif", "Umer Sharif comedian", "m", 0),
    ("junaid_jamshed", "Junaid Jamshed", "m", 0),
    ("jawed_sheikh", "Jawed Sheikh actor", "m", 0),
    ("qavi_khan", "Qavi Khan actor", "m", 0),
    ("zia_mohyeddin", "Zia Mohyeddin", "m", 0),
]

EVAL_EXCLUDE = {
    "ahsan_iqbal", "ali_amin_gandapur", "asad_umar", "asif_ali_zardari",
    "asim_munir", "benazir_bhutto", "bilawal_bhutto_zardari", "fawad_chaudhry",
    "fazl_ur_rehman", "hamza_shehbaz", "hina_rabbani_khar", "imran_khan",
    "khawaja_asif", "maryam_nawaz_sharif", "murad_ali_shah", "nawaz_sharif",
    "pervez_elahi", "pervez_musharraf", "qamar_javed_bajwa", "shah_mehmood_qureshi",
    "shehbaz_sharif", "sheikh_rashid_ahmed", "yousaf_raza_gillani",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-id", type=int, default=45)
    ap.add_argument("--only", default="", help="comma-separated slugs")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    overlap = {s for s, *_ in PEOPLE} & EVAL_EXCLUDE
    if overlap:
        sys.exit(f"REFUSING: training identities overlap the eval set: {sorted(overlap)}")

    people = PEOPLE
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        people = [p for p in PEOPLE if p[0] in wanted]
        missing = wanted - {p[0] for p in people}
        if missing:
            sys.exit(f"unknown slugs: {sorted(missing)}")

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"collecting {len(people)} identities x {args.per_id} images -> {OUT}", flush=True)

    done = 0
    for slug, query, gender, veiled in people:
        dest = OUT / slug
        existing = len(list(dest.glob("*"))) if dest.exists() else 0
        if existing >= args.per_id and not args.overwrite:
            print(f"[{done+1}/{len(people)}] {slug}: {existing} already present", flush=True)
            done += 1
            continue
        dest.mkdir(parents=True, exist_ok=True)
        # icrawler writes 000001.jpg ... and resumes from the existing count
        crawler = BingImageCrawler(
            feeder_threads=1, parser_threads=1, downloader_threads=2,
            storage={"root_dir": str(dest)},
        )
        try:
            crawler.crawl(
                keyword=query,
                max_num=args.per_id,
                min_size=(120, 120),
                file_idx_offset="auto",
            )
        except Exception as exc:  # network hiccups should not kill the run
            print(f"  ! {slug}: {exc!r}", flush=True)
        n = len(list(dest.glob("*")))
        done += 1
        print(f"[{done}/{len(people)}] {slug:<22s} {n:3d} imgs  ({query})", flush=True)

    rows = []
    for slug, query, gender, veiled in people:
        n = len(list((OUT / slug).glob("*"))) if (OUT / slug).exists() else 0
        rows.append({"identity": slug, "query": query, "gender": gender,
                     "veiled": veiled, "raw_images": n})
    with open(OUT / "sources.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nmanifest -> {OUT / 'sources.csv'}  "
          f"({sum(r['raw_images'] for r in rows)} images, "
          f"{sum(1 for r in rows if r['raw_images'])} identities)", flush=True)


if __name__ == "__main__":
    main()
