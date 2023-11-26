import numpy as np
import random
random.seed(42)
np.random.seed(42)


def load_files(year, max_files):
    times_cao = np.load("/uio/hume/student-u37/fslippe/data/dates_for_labeling/cao_thr30_%s_times.npy" %(year))[:max_files]
    dates_cao = np.load("/uio/hume/student-u37/fslippe/data/dates_for_labeling/cao_thr30_%s_dates.npy" %(year))[:max_files]
    dates_nocao = np.load("/uio/hume/student-u37/fslippe/data/dates_for_labeling/nocao_thr2_%s_dates.npy" %(year))[:max_files]
    times_nocao = np.load("/uio/hume/student-u37/fslippe/data/dates_for_labeling/nocao_thr2_%s_times.npy" %(year))[:max_files]
    times_random = np.load("/uio/hume/student-u37/fslippe/data/dates_for_labeling/random_sample_8-14_%s_times.npy" %(year))
    dates_random = np.load("/uio/hume/student-u37/fslippe/data/dates_for_labeling/random_sample_8-14_%s_dates.npy" %(year))

    return dates_cao, times_cao, dates_nocao, times_nocao, dates_random, times_random

def find_new_random(dates_cao, times_cao, dates_nocao, times_nocao, dates_random, times_random, max_files):
    d = np.append(dates_cao, dates_nocao)
    t = np.append(times_cao, times_nocao)

    full_list = list(zip(d,t))

    new_dates_random = []
    new_times_random = []

    for i in range(len(dates_random)):
        if (dates_random[i], times_random[i]) not in full_list:
            new_dates_random.append(dates_random[i])
            new_times_random.append(times_random[i])
        if len(new_dates_random) == max_files:
            break

    assert len(new_dates_random) == len(new_times_random) == max_files

    return new_dates_random, new_times_random

def handle_years(years):
    dates_cao_tot = []
    times_cao_tot = []
    dates_nocao_tot = []
    times_nocao_tot = []
    dates_random_tot = []
    times_random_tot = []
    dates_tot = []
    times_tot = []

    for year in years:
        max_files = 12 if year != 2023 else 8

        dates_cao, times_cao, dates_nocao, times_nocao, dates_random, times_random = load_files(year, max_files)

        new_dates_random, new_times_random = find_new_random(dates_cao, times_cao, dates_nocao, times_nocao, dates_random, times_random, max_files)

        dates_cao_tot.extend((dates_cao))
        times_cao_tot.extend((times_cao))
        dates_nocao_tot.extend((dates_nocao))
        times_nocao_tot.extend((times_nocao))
        dates_random_tot.extend((new_dates_random))
        times_random_tot.extend((new_times_random))   
        dates_tot.extend((dates_cao))
        dates_tot.extend((dates_nocao))
        dates_tot.extend((new_dates_random)) 
        times_tot.extend((times_cao))
        times_tot.extend((times_nocao))
        times_tot.extend((new_times_random))

    return dates_cao_tot, times_cao_tot, dates_nocao_tot, times_nocao_tot, dates_random_tot, times_random_tot, dates_tot, times_tot

def create_blocks(dates_cao_tot, times_cao_tot, dates_nocao_tot, times_nocao_tot, dates_random_tot, times_random_tot):
    fourth = int(len(dates_cao_tot)/4)

    indices = random.sample(range(len(dates_cao_tot)), fourth)

    dates_cao_block = [dates_cao_tot[i] for i in indices]
    times_cao_block = [times_cao_tot[i] for i in indices]

    dates_nocao_block = [dates_nocao_tot[i] for i in indices]
    times_nocao_block = [times_nocao_tot[i] for i in indices]

    dates_random_block = [dates_random_tot[i] for i in indices]
    times_random_block = [times_random_tot[i] for i in indices]

    dates_block = dates_cao_block + dates_nocao_block + dates_random_block
    times_block = times_cao_block + times_nocao_block + times_random_block
    
    # remove selected indices from total
    for i in sorted(indices, reverse=True):
        del dates_cao_tot[i]
        del times_cao_tot[i]
        del dates_nocao_tot[i]
        del times_nocao_tot[i]
        del dates_random_tot[i]
        del times_random_tot[i]

    return dates_block, times_block, dates_cao_tot, times_cao_tot, dates_nocao_tot, times_nocao_tot, dates_random_tot, times_random_tot


def main():
    years = [2019, 2020, 2021, 2022, 2023]
    dates_cao_tot, times_cao_tot, dates_nocao_tot, times_nocao_tot, dates_random_tot, times_random_tot, dates_tot, times_tot = handle_years(years)
    dates_block, times_block, dates_cao_tot, times_cao_tot, dates_nocao_tot, times_nocao_tot, dates_random_tot, times_random_tot = create_blocks(dates_cao_tot, times_cao_tot, dates_nocao_tot, times_nocao_tot, dates_random_tot, times_random_tot)

    paired_data = list(zip(dates_block, times_block))

    random.shuffle(paired_data)

    dates_block, times_block = zip(*paired_data)
    dates_tot = dates_cao_tot + dates_nocao_tot + dates_random_tot
    times_tot = times_cao_tot + times_nocao_tot + times_random_tot

    paired_data = list(zip(dates_tot, times_tot))

    random.shuffle(paired_data)

    dates_tot, times_tot = zip(*paired_data)

    np.save("/uio/hume/student-u37/fslippe/data/dates_for_labeling/day_filtered/dates_block", dates_block)
    np.save("/uio/hume/student-u37/fslippe/data/dates_for_labeling/day_filtered/times_block", times_block)

    np.save("/uio/hume/student-u37/fslippe/data/dates_for_labeling/day_filtered/dates_rest", dates_tot)
    np.save("/uio/hume/student-u37/fslippe/data/dates_for_labeling/day_filtered/times_rest", times_tot)

if __name__ == "__main__":
    main()

