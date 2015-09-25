# Example: prints statistics.
#
import pyinotify
import sys
import os
import random
import PIL.Image
import string
import subprocess

WATCH_FILE = '/home/vj/dbsh/code/scan_me'
FILES = []

class Identity(pyinotify.ProcessEvent):
    def process_file(self, file):
        ext = file.split(".")[-1]
        if not ext.lower() in ["bmp", "jpeg", "jpg", "bmp", "png"]:
            return
        if os.path.isfile(WATCH_FILE + "/" + file.split(".")[0] + ".txt"):
            print "skipping " + file
            return

        print "would process " + WATCH_FILE + "/" + file
        # start subprocess, stdout goes here
        # file written to where we need it
        # system("python foo.py &")
        # add Jon's changes
        myenv = os.environ
        myenv["PYTHONPATH"] = "/home/vj/dbhw/caffe/python:" + myenv["PYTHONPATH"]
        subprocess.Popen(["/home/vj/dbsh/code/exec_file.sh", WATCH_FILE + "/" + file], env=myenv)
        
    def process_default(self, event):
        for (dirpath, dirnames, filenames) in os.walk(WATCH_FILE):
            for file in filenames:
                if not FILES.__contains__(file):
                    FILES.append(file)
                    self.process_file(file)

def on_loop(notifier):
    # notifier.proc_fun() is Identity's instance
    s_inst = notifier.proc_fun().nested_pevent()
    print repr(s_inst), '\n', s_inst, '\n'


if __name__ == "__main__":
    wm = pyinotify.WatchManager()
    # Stats is a subclass of ProcessEvent provided by pyinotify
    # for computing basics statistics.
    s = pyinotify.Stats()
    notifier = pyinotify.Notifier(wm, default_proc_fun=Identity(s), read_freq=3)
    wm.add_watch(WATCH_FILE, pyinotify.IN_CREATE | pyinotify.IN_MOVED_FROM | pyinotify.IN_MOVED_TO | pyinotify.IN_MOVE_SELF | pyinotify.IN_CLOSE_WRITE, rec=True, auto_add=True)
    notifier.loop(callback=on_loop)
