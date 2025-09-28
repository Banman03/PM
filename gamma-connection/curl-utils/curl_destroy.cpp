#include <curl/curl.h>


void curl_destroy(CURL* curl_handler_main) {
    curl_easy_cleanup(curl_handler_main);
}