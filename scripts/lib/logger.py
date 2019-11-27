# logger module

from datetime import datetime
import os

logfile = None
logbuf = []

def init(analysis_path):
    global logfile
    if not os.path.isdir(analysis_path):
        log("logger: analysis_path missing:", analysis_path)
    use_log_dir = False
    if use_log_dir:
        logdir = os.path.join(analysis_path, "log")
        if not os.path.isdir(logdir):
            log("logger: creating log directory:", logdir)
            os.makedirs(logdir)
        logfile = os.path.join(logdir, "messages")
    else:
        logfile = os.path.join(analysis_path, "messages")

def log(*args, quiet=False):
    global logbuf
    # assemble message line
    msg = []
    now = datetime.now()
    msg.append(str(now) + ":")
    for a in args:
        msg.append(str(a))
    logbuf.append(" ".join(msg))
    if logfile:
        # flush log buffer
        f = open(logfile, "a")
        for line in logbuf:
            f.write(line)
            f.write("\n")
        f.close()
        logbuf = []
    if not quiet:
        print(*msg)

# log quietly (log to file, but not to stdout)
def qlog(*args):
    log(*args, quiet=True)
    
