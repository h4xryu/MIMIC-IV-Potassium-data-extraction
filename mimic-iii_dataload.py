from torch.utils.data import Dataset, DataLoader
from utils.dataset.Severance import *
from utils.one_cycle_ecg import *
import torch
from scipy.signal import resample


#########################################################################################################
#                                            MIMIC-III                                                  #
#########################################################################################################
import pandas as pd
import numpy as np
import wfdb
from pathlib import Path
import re
from datetime import datetime
from tqdm import tqdm


class MIMIC3CSVMatcher:
    def __init__(self, wdb_path, csv_path):
        """
        Initialize matcher for MIMIC-III using CSV files
        Args:
            wdb_path (str): Path to MIMIC-III waveform matched database
            csv_path (str): Path to MIMIC-III CSV files directory
        """
        self.wdb_path = Path(wdb_path)
        self.csv_path = Path(csv_path)

    def load_clinical_data(self):
        """Load required tables from CSV files"""
        print("Loading clinical data...")

        # Load required tables
        chartevents = pd.read_csv(self.csv_path / 'CHARTEVENTS.csv')
        icustays = pd.read_csv(self.csv_path / 'ICUSTAYS.csv')
        admissions = pd.read_csv(self.csv_path / 'ADMISSIONS.csv')

        # Convert time columns to datetime
        time_columns = ['CHARTTIME', 'INTIME', 'OUTTIME', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME']
        for df in [chartevents, icustays, admissions]:
            for col in time_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])

        return chartevents, icustays, admissions

    def get_potassium_data(self, chartevents, icustays, admissions):
        """Get potassium measurements with ICU stay information and K+ level labels"""
        print("Processing potassium measurements...")

        # Filter for potassium measurements
        potassium_itemids = [227442]  # POTASSIUM serum
        potassium_labs = chartevents[
            (chartevents['ITEMID'].isin(potassium_itemids)) &
            (chartevents['VALUENUM'].notna()) &
            (chartevents['VALUENUM'] > 0) &
            (chartevents['VALUENUM'] <= 30)
            ]

        # Merge with ICU stays
        merged_data = pd.merge(
            potassium_labs,
            icustays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME']],
            on=['SUBJECT_ID', 'HADM_ID'],
            how='inner'
        )

        # Filter measurements during ICU stay (with 6 hours before)
        merged_data = merged_data[
            (merged_data['CHARTTIME'] >= merged_data['INTIME'] - pd.Timedelta(hours=6)) &
            (merged_data['CHARTTIME'] <= merged_data['OUTTIME'])
            ]

        # Add mortality information
        merged_data = pd.merge(
            merged_data,
            admissions[['SUBJECT_ID', 'HADM_ID', 'DEATHTIME']],
            on=['SUBJECT_ID', 'HADM_ID'],
            how='left'
        )

        # Calculate ICU mortality
        merged_data['MORT_ICU'] = (
                (merged_data['DEATHTIME'].notna()) &
                (merged_data['DEATHTIME'] >= merged_data['INTIME']) &
                (merged_data['DEATHTIME'] <= merged_data['OUTTIME'])
        ).astype(int)

        # Add potassium level labels
        def label_k_level(k_value):
            if k_value < 3.5:
                return 'hypokalemia'
            elif k_value >= 5.5:
                return 'hyperkalemia'
            else:
                return 'normal'

        merged_data['k_level_label'] = merged_data['VALUENUM'].apply(label_k_level)

        # Add numeric labels for machine learning
        k_level_map = {'hypokalemia': 0, 'normal': 1, 'hyperkalemia': 2}
        merged_data['k_level_numeric'] = merged_data['k_level_label'].map(k_level_map)

        return merged_data

    def parse_record_info(self, record_path):
        """Parse subject ID and datetime from record path"""
        subject_match = re.search(r'p00*(\d+)', record_path[3:])
        subject_id = int(subject_match.group(1)) if subject_match else None

        datetime_match = re.search(r'-(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})', record_path)
        if datetime_match:
            year, month, day, hour, minute = map(int, datetime_match.groups())
            record_datetime = datetime(year, month, day, hour, minute)
        else:
            record_datetime = None

        return subject_id, record_datetime

    def load_waveform_data(self, record_path, window_minutes=1):
        """Load waveform data with specified window"""
        try:
            record = wfdb.rdrecord(
                record_name=str(self.wdb_path / record_path.strip())
            )

            # 필요한 리드 정의
            required_leads = ['II']

            # 헤더의 리드 이름을 대문자로 변환하여 비교
            available_leads = [lead.upper() for lead in record.sig_name]

            # 모든 필요한 리드가 있는지 확인
            if not all(lead in available_leads for lead in required_leads):
                return None, None, None

            # 필요한 리드의 인덱스 찾기
            lead_indices = [available_leads.index(lead) for lead in required_leads]

            # Extract window of data
            samples_needed = int(window_minutes * 60 * record.fs)
            mid_point = len(record.p_signal) // 2
            start_idx = max(0, mid_point - samples_needed // 2)
            end_idx = min(len(record.p_signal), mid_point + samples_needed // 2)

            signals = record.p_signal[start_idx:end_idx, lead_indices]

            return signals, record.fs, required_leads

        except Exception as e:
            print(f"Error loading record {record_path}: {str(e)}")
            return None, None, None
            return signals, record.fs, available_leads

        except Exception as e:
            print(f"Error loading record {record_path}: {str(e)}")
            return None, None, None

    def find_matching_records(self, time_threshold_hours=1):
        """Find matching waveform records for potassium measurements"""
        # Load clinical data
        labevents, icustays, admissions = self.load_clinical_data()
        potassium_data = self.get_potassium_data(labevents, icustays, admissions)

        # Get all records
        with open(self.wdb_path / 'RECORDS-waveforms', 'r') as f:
            records = [line.strip() for line in f.readlines()]

        matches = []

        # Create dictionary of records by subject_id for faster lookup
        print("Organizing waveform records...")
        records_by_subject = {}
        for record in records:
            subject_id, record_time = self.parse_record_info(record)
            if subject_id not in records_by_subject:
                records_by_subject[subject_id] = []
            records_by_subject[subject_id].append((record, record_time))

        # Find matches
        print("Finding matches...")
        for _, row in tqdm(potassium_data.iterrows()):
            subject_id = row['SUBJECT_ID']
            lab_time = row['CHARTTIME']

            if subject_id not in records_by_subject:
                continue

            # Find closest record in time
            subject_records = records_by_subject[subject_id]
            closest_record = None
            min_time_diff = float('inf')

            for record, record_time in subject_records:
                time_diff = abs((pd.to_datetime(lab_time) - record_time).total_seconds() / 3600)
                if time_diff < time_threshold_hours and time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_record = record

            if closest_record:
                # Load waveform data
                signals, fs, leads = self.load_waveform_data(closest_record)
                if signals is not None:
                    matches.append({
                        'subject_id': subject_id,
                        'record_path': closest_record,
                        'Potassium (serum)': row['VALUENUM'],
                        'k_level_label': row['k_level_label'],
                        'k_level_numeric': row['k_level_numeric'],
                        'mort_icu': row['MORT_ICU'],
                        'time_diff_hours': min_time_diff,
                        'signals': signals,
                        'fs': fs,
                        'leads': leads,
                        'lab_time': lab_time
                    })

        return matches


def main():
    # Initialize matcher with your paths
    wdb_path = "/media/jwlee/9611c7a0-8b37-472c-8bbb-66cac63bc1c7/downloads/physionet.org/files/mimic3wdb-matched/1.0"
    csv_path = "/media/jwlee/9611c7a0-8b37-472c-8bbb-66cac63bc1c7/Hyperkalemia진단모델/dataset_mimic/mimic_iii_1_4"

    matcher = MIMIC3CSVMatcher(wdb_path, csv_path)

    # Find matches
    matches = matcher.find_matching_records(time_threshold_hours=2)

    # Print summary
    print(f"\nFound {len(matches)} matching records")

    if matches:
        # Analyze time differences
        time_diffs = [m['time_diff_hours'] for m in matches]
        print(f"Average time difference: {np.mean(time_diffs):.2f} hours")

        # Analyze potassium distributions and labels
        k_labels = [m['k_level_label'] for m in matches]
        label_counts = {
            'hypokalemia': k_labels.count('hypokalemia'),
            'normal': k_labels.count('normal'),
            'hyperkalemia': k_labels.count('hyperkalemia')
        }

        print("\nPotassium Level Distribution:")
        for label, count in label_counts.items():
            print(f"{label}: {count} cases ({count / len(k_labels) * 100:.1f}%)")

        # Save matches with labels
        np.save('matched_data_with_labels.npy', matches)


if __name__ == "__main__":
    main()
