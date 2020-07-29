#!/usr/bin/python
# -*- coding: UTF-8 -*-




'''
we use libcurl to do http get and post
pycurl is a python package for libcurl
'''


'''
if we want to upload a file by http

the form must be defined like this
<!DOCTYPE html>
<title>Upload File</title>
<h1>Upload File</h1>
<form method=post enctype=multipart/form-data>
     <input type=file name=file>
     <input type=submit value=upload>
</form>

we can use curl to upload like the html form
curl -F "file=@snapshot.jpg" "http://127.0.0.1:5000/"
'''




import pycurl



# upload a file
def upload_file(url,html_input_box_name,upload_file_path):
    c = pycurl.Curl()
    c.setopt(c.POST, 1)
    c.setopt(c.URL, url)
    c.setopt(c.HTTPPOST, [(html_input_box_name, (c.FORM_FILE, upload_file_path))])
    c.perform()
    c.close()











