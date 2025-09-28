#include <stdio.h>
#include <curl/curl.h>
#include <boost/thread.hpp>
#include "curl_utils.hpp";

int main(int argc, char** argv) {
    CURL* curl_handler_main;
    CURLcode* curl_setup_response_main;

    curl_init(curl_handler_main, curl_setup_response_main);

    

    curl_destroy(curl_handler_main);

    return 0;
}