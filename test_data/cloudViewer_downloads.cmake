include(ExternalProject)

set(DOWNLOAD_DIR "${CMAKE_CURRENT_LIST_DIR}")

ExternalProject_Add(
    ext_cloudViewer_downloads
    PREFIX cloudViewer_downloads
    URL https://github.com/Asher-1/cloudViewer_downloads.git
    URL_HASH SHA256=2f5f2b789edb00260aa71f03189da5f21cf4b5617c4fbba709e9fbcfc76a2f1e
    DOWNLOAD_DIR "${DOWNLOAD_DIR}"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_cloudViewer_downloads SOURCE_DIR)
set(TEST_DATA_DIR ${SOURCE_DIR})
