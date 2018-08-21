import numpy as np

def getSessionClocks(fname):
    """Read the timestamps of the clock as EEG sample numbers
    The result is split into a list of clock-timestamps for every individual
    session

    Args:
        fname (str): Filename of the file containing the markers without
            the filename extension. '.vmrk' is added to fname

    Returns:
        session_clocks (list): list with arrays of indices to the EEG
            signaling the occurence of a clock trigger input such that
            session_clocks[0] contains the timestamps of the clock for the
                1st session
            session_clocks[1] contains the timestamps of the clock for the
                2nd session
                ...
    """
    marker = np.loadtxt(fname, skiprows=12, usecols=2,
            delimiter=',', dtype=int)
    # split into sessions
    diff = np.diff(marker)
    session_start = np.r_[marker[0], marker[1:][diff > 1005]]
    session_clocks = [marker[np.all([marker>=s, marker<e], axis=0)]
            for s, e in
            zip(session_start, np.r_[session_start[1:], np.inf])]
    return session_clocks

def SyncMusicToEEG(eeg_session_clocks, nearest_session_clocks,
        session_devs, s_rate=1000):
    """Synchronize events of audio files to EEG

    Args:
        eeg_session_clocks (list of ndarrays): a list of array (one array
            for every session) containing the sample indices of clock ticks
        nearest_session_clocks (list of ndarrays): a list of array (one
            array per session) containing the index of the nearest clock
            of that session (in every session the clocks are counted
            starting from 0)
        session_devs (list of ndarrays): the relative latencies (in s) of
            the events in the audio files to the clocks given in 
            nearest_session_clocks

    Returns:
        event_marker (ndarray of ints): the sample indices relative to the
            EEG of all the events in session_dev
    """
    cue_pos = np.hstack([e[m] + np.round(d*s_rate).astype(int) for e,m,d in
        zip(eeg_session_clocks, nearest_session_clocks, session_devs)])
    return cue_pos

