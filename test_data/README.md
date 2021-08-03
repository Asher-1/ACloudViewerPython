# CloudViewer-Downloads

Hosting CloudViewer test data for development use.

## (TL;DR) How to add a data file

- Small files (e.g. several MB)
    - Step 1: Push the file to the `cloudViewer_downloads` repo.
    - Step 2: Get the **direct** download URL to the file. The URL typically
      looks like `https://github.com/Asher-1/cloudViewer_downloads/raw/master/path/to/file`.
      You may get the URL by going to the file in the GitHub website, right
      click on the "Download" button and select "Copy link address".
    - Step 3: Back to the `CloudViewer` repo, edit
      `CloudViewer/examples/test_data/download_file_list.json` to specify the
      direct download URL and download path.
    - Step 4: Done! When you re-build the `CloudViewer` project, the new file will
      be downloaded to `CloudViewer/examples/test_data/cloudViewer_downloads`. Now you
      can use this file in your source code including unit tests.
- Large files (max 2GB)
    - Step 1: Create a new release or re-use an existing release in the
      `cloudViewer_downloads` repo.
    - Step 2: Upload your file as a release artifact. For files larger than 2GB,
      you may split the files into parts with tools like `zip` or `tar`. After
      uploading, you will able to get the direct download URL.
    - Step 3: These files will not be downloaded automatically. You need to
      add your own mechanisms to download and consume the files.

## When is the download triggered

Files listed in `CloudViewer/examples/test_data/download_file_list.json` will be
downloaded automatically in the following scenarios:

- when running CMake config steps
- when running Python unit tests
- when running Python examples/tutorials

## Permission control

Community developers can create pull requests to add new files to the
`cloudViewer_downloads` repo. Internal developers can directly push to the
`cloudViewer_downloads` repo's `master` branch and create/modify releases.
