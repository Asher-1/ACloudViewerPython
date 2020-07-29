import os
import logging
import multiprocessing

bind = '0.0.0.0:9995'
# workers = multiprocessing.cpu_count() * 2 + 1
workers = 4
print('*' * 50 + 'the number of workers is : {} worker'.format(workers), '*' * 50)
backlog = 2048
worker_class = "gevent"
daemon = True
debug = False
reload = True
chdir = ""
print("chdir : {}".format(chdir))
proc_name = 'gunicorn.proc'

loglevel = 'debug'
access_log_format = '%(t)s %(p)s %(h)s "%(r)s" %(s)s %(L)s %(b)s %(f)s" "%(a)s"'
accesslog = os.path.join(chdir, "log", "gunicorn_access.log")
errorlog = os.path.join(chdir, "log", "gunicorn_error.log")
