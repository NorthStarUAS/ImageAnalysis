# logger module

from datetime import datetime
import os
import socket                   # gethostname()

logfile = None
logbuf = []

def init(analysis_path):
    global logfile
    if not os.path.isdir(analysis_path):
        log("logger: analysis_path missing:", analysis_path)
    logfile = os.path.join(analysis_path, "messages-" + socket.gethostname())

# log a message to messages files (and to stdout by default)
def log(*args, quiet=False, fancy=False):
    global logbuf
    # timestamp
    now = datetime.now()
    timestamp = str(now) + ": "
    # assemble message line
    msg = []
    for a in args:
        msg.append(str(a))
    if not fancy:
        logbuf.append(timestamp + " ".join(msg))
    else:
        logbuf.append("")
        logbuf.append("############################################################################")
        logbuf.append("### " + timestamp + " ".join(msg))
        logbuf.append("############################################################################")
        logbuf.append("")
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
