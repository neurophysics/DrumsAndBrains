import numpy as np
import aifc
import matplotlib.pyplot as plt
import sys
import os.path

# check that all files have the equal number of samples
def check_length(*fnames):
    fsize = []
    for fname in fnames:
        f = aifc.open(fname, 'r')
        fsize.append(f.getnframes())
    assert np.all(np.asarray(fsize)==fsize[0])

# define the funtions to get timings of all beats and to extract cues
def moving_average(a, n):
    """
    Calculate the moving average of a sequence a across n samples
    the ends are mirrored to ameliorate edge effects

    Notes:
    Needed for normalizing the noise level of the syncIn recordings
    """
    # mirror the edges
    a = np.r_[a[1:n//2 + 1][::-1], a, a[-n//2: - 1][::-1]]
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def get_ClickTime(fname, thresh=2000, mindiff=0.1, normalize=False):
    """
    Read data from file and extract the timing of all signals

    Args:
        fname (str): the filename of an audio file
            is expected to be an uncompressed 16bit mono aifc file
        thresh (int): threshold to extract click signals, defaults to 1000
        mindiff (float): minimal latency difference (in seconds) that is
            expected to be between consecutive click signals
        normalize (bool): if true, the noise level is normalized
            needed for the syncIn files because they com with different
            gain settings
    Returns:
        threshtimes (ndarray): an array of the timings of clicks in s
    """
    f = aifc.open(fname, 'r')
    data = np.frombuffer(f.readframes(f.getnframes()), dtype='>i2')
    # check that the number of entries equals the number of samples
    assert len(data) == f.getnframes()
    if normalize:
       # calculate the moving standard deviation across 1s
       data_mean = moving_average(data, f.getframerate())
       data_sd = np.sqrt(moving_average((data - data_mean)**2,
               f.getframerate()))
       # normalize
       data_sd[data_sd < 0.1*np.mean(data_sd)]=np.mean(data_sd)
       data = data/data_sd
    threshtime = (np.abs(data)>thresh).nonzero()[0]/float(f.getframerate())
    f.close()
    while not np.all(np.diff(threshtime) >= mindiff):
        threshtime = np.r_[threshtime[0], threshtime[1:][
            np.diff(threshtime)>mindiff]]
    return threshtime

def GetCueTime(clicktimes, mindiff):
    # a click is recognized as cue if the interval to the previous
    # and following click is atleast mindiff
    return clicktimes[1:-1][
            ((clicktimes[2:] - clicktimes[1:-1]) > mindiff) &
            ((clicktimes[1:-1] - clicktimes[:-2]) > mindiff)]

def GetAllCues(snareStim_times, wdBlkStim_times, mindiff=0.9):
    """Get the timings of the cues for both instruments

    Input:
        snareStim_times (ndarray): array of all timings for the snare
            drum stimulus (duple rhythm)
        wdBlkStim_times (ndarray): array of all timings for the woodblock
            stimulus (triple rhythm)
        mindiff (float): the minimal latency difference expected to
            be between every beat of the stimulus instruments and a cue
    Returns:
        snareCue_times (ndarray): array of all timings for the snare drum
            cues
        wdBlkCue_times (ndarray): array of all timings for the woodblock
            cues
    """
    snareCue_times = GetCueTime(snareStim_times, mindiff)
    wdBlkCue_times = GetCueTime(wdBlkStim_times, mindiff)
    # the last beat of either stimuli is always a cue
    # however it is not followed by a beat of the same instrument and
    # therefor not detected by GetCueTime() - append it now
    if snareStim_times[-1]>wdBlkStim_times[-1]:
        snareCue_times = np.r_[snareCue_times, snareStim_times[-1]]
    else:
        wdBlkCue_times = np.r_[wdBlkCue_times, wdBlkStim_times[-1]]
    return snareCue_times, wdBlkCue_times

def get_ResponseLatency(cue_time, drum_time, bar_duration):
    """Calculate the latency between cue and response
    The reponse is counted if it is whithin the two bars following the cue.
    When several responses occured, the first is chose.

    Input:
        cue_time (ndarray): the timings of the cue
        drum_time (ndarray): the timings of the response
        bar_length (float): the length of a bar

    Output:
        latency (numpy.ma.narray): the latency between cues and responses,
            if no response was recorded, the element is masked
    """
    candidate_beat_latencies = [
            drum_time[(drum_time>c) & (drum_time<(c+2*bar_duration))]
            for c in cue_time]
    latencies = [l[0] - c if len(l)>0 else np.nan for l,c in zip(
        candidate_beat_latencies, cue_time)]
    return np.ma.masked_invalid(latencies)

# seperate clock into sessions and measure deviation to closest clock of
# session
def getSessionClocks(fname, thresh=25000):
    syncIn_times = get_ClickTime(fname, thresh=thresh)
    # whenever clicks are seperated by more than 1s, recognize as new
    # session
    diff = np.diff(syncIn_times)
    session_start = np.r_[syncIn_times[0], syncIn_times[1:][diff > 1.005]]
    session_clocks = [syncIn_times[
        np.all([syncIn_times>=s, syncIn_times<e],
            axis=0)]
        for s, e in zip(session_start, np.r_[session_start[1:], np.inf])]
    return session_clocks

def DevFromNearestClock(clocks, t):
    diff = np.diff(clocks)
    session_start = np.r_[0, (diff > 1.005).nonzero()[0] + 1]
    # get closest neighbour clock of every t
    nearestClock = np.abs(t[:,np.newaxis] - clocks).argmin(axis=1)
    nearestDev =(t[:,np.newaxis] - clocks)[range(len(t)),nearestClock]
    # find out to which session the clock belongs and re-index such that
    # each session starts with clock 0
    nearest_sessionClock = [
            nearestClock[(nearestClock>=s) & (nearestClock<e)] - s
            for s, e in zip(session_start, np.r_[session_start[1:], np.inf])]
    nearest_sessionDev = [
            nearestDev[(nearestClock>=s) & (nearestClock<e)]
            for s, e in zip(session_start, np.r_[session_start[1:], np.inf])]
    return nearest_sessionClock, nearest_sessionDev

if __name__=="__main__" :
    data_folder = sys.argv[1]
    subject = int(sys.argv[2])
    result_folder = sys.argv[3]

    data_folder = os.path.join(data_folder, 'S%02d' % subject)
    save_folder = os.path.join(result_folder, 'S%02d' % subject)

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    wdBlkStim_fname = os.path.join(data_folder,
            'NeuralEntrStimLive8_Ss%02d_WdBlkStim.aif' % subject)
    snareStim_fname =  os.path.join(data_folder,
            'NeuralEntrStimLive8_Ss%02d_SnareStim.aif' % subject)
    syncIn_fname =  os.path.join(data_folder,
            'NeuralEntrStimLive8_Ss%02d_SyncIn.aif' % subject)
    drumIn_fname =  os.path.join(data_folder,
            'NeuralEntrStimLive8_Ss%02d_DrumIn.aif' % subject)

    check_length(wdBlkStim_fname, snareStim_fname, syncIn_fname, drumIn_fname)

    # read the click times of the instruments
    wdBlkStim_times = get_ClickTime(wdBlkStim_fname)
    snareStim_times = get_ClickTime(snareStim_fname)

    # get the cue positions - as mindiff, choose half a bar plus 50 ms
    snareCue_times, wdBlkCue_times = GetAllCues(
            snareStim_times, wdBlkStim_times, mindiff=bar_duration/2. + 0.05)

    # check that we indeed have all the 75 stimuli
    assert len(snareCue_times) == 75
    assert len(wdBlkCue_times) == 75

    # read the drum response
    drum_times = get_ClickTime(drumIn_fname)

    # calculate the latency between cue and response
    snare_latencies = get_ResponseLatency(snareCue_times, drum_times,
            2*bar_duration)
    wdBlk_latencies = get_ResponseLatency(wdBlkCue_times, drum_times,
            2*bar_duration)

    # calculate the deviation to the intended beat latency
    snare_deviation = snare_latencies - bar_duration/2.
    wdBlk_deviation = wdBlk_latencies - 2*bar_duration/3

    # calculate the absolute precision
    snare_precision = np.abs(snare_deviation)
    wdBlk_precision = np.abs(wdBlk_deviation)

    hist_bins = np.arange(0.5, 1.5 + 0.025, 0.025)

    # plot the results
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.hist(snare_latencies, bins=hist_bins, color='b', label='duple cue',
            edgecolor='w', alpha=0.6)
    ax1.axvline(bar_duration/2., color='b', label='correct duple lat.')
    ax1.hist(wdBlk_latencies, bins=hist_bins, color='r', label='triple cue',
            edgecolor='w', alpha=0.6)
    ax1.axvline(2*bar_duration/3., color='r', label='correct triple lat.')
    ax1.set_xlabel('latency to cue (s)')
    ax1.set_ylabel('number of trials')
    ax1.legend(loc='upper left')
    ax1.set_ylim([0,30])
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(save_folder,
        'NeuralEntrl_Ss%02dresponse.png' % subject))
    fig.savefig(os.path.join(save_folder,
        'NeuralEntrl_Ss%02dresponse.pdf' % subject))

    syncIn_times = get_ClickTime(syncIn_fname, thresh=13, normalize=True)

    snareCue_nearestClock, snareCue_DevToClock = DevFromNearestClock(
            syncIn_times, snareCue_times)
    wdBlkCue_nearestClock, wdBlkCue_DevToClock = DevFromNearestClock(
            syncIn_times, wdBlkCue_times)

    # save the results
    np.savez(os.path.join(save_folder, 'behavioural_results.npz'),
        snareCue_nearestClock = np.array(snareCue_nearestClock, dtype=object),
        snareCue_DevToClock = np.array(snareCue_DevToClock, dtype=object),
        wdBlkCue_nearestClock = np.array(wdBlkCue_nearestClock, dtype=object),
        wdBlkCue_DevToClock = np.array(wdBlkCue_DevToClock, dtype=object),
        snareCue_times = snareCue_times,
        wdBlkCue_times = wdBlkCue_times,
        bar_duration = bar_duration,
        snare_deviation = snare_deviation.data,
        wdBlk_deviation = wdBlk_deviation.data,
        )

    # plot self performance
    self_perf = []
    self_vigil = []
    with open(os.path.join(data_folder,'S%02d_self-assessment.txt' % subject)) as f:
        for line in f:
            l = line.split(' ')
            if line[0] == 'P':
                self_perf.append(int(l[1]))
            if line[0] == 'V':
                self_vigil.append(int(l[1]))

    fig = plt.figure()
    meandevs_snare = [np.mean(np.abs(s)) for s in snareCue_DevToClock]
    meandevs_wdBlk = [np.mean(np.abs(s)) for s in wdBlkCue_DevToClock]
    corrSnare = np.corrcoef(meandevs_snare, self_perf)[0][1]
    corrWdBlk = np.corrcoef(meandevs_wdBlk, self_perf)[0][1]
    plt.plot(meandevs_snare, marker = 'o', label = 'abs mean dev snare')
    plt.plot(meandevs_wdBlk, marker = 'o', label = 'abs mean dev wdBlk')
    plt.plot([(10.-s)/10 for s in self_perf], marker = 'o', label = '10-Performance / 10')
    plt.plot([(10.-s)/10 for s in self_vigil], marker = 'o', label = '10-Vigilance / 10')
    plt.legend()
    plt.ylim(0.,0.8)
    plt.xlabel('sessions (~25 trials each)')
    plt.ylabel('absolute deviation / self assessment score')
    plt.title('self assessment vs snare/wdblk performance (r = %02f / %02f)'% (corrSnare, corrWdBlk))
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(save_folder,
        'SelfAssessmentAndResponseSs%02d.png' % subject))
    fig.savefig(os.path.join(save_folder,
        'SelfAssessmentAndResponseSs%02d.pdf' % subject))
