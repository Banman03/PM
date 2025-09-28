#include <stdio.h>
#include <curl/curl.h>

void curl_init(CURL* curl_handler_main, CURLcode* curl_setup_response_main) {
    curl_global_init(CURL_GLOBAL_SSL);
    curl_handler_main = curl_easy_init();
    *curl_setup_response_main = curl_easy_setopt(curl_handler_main, CURLOPT_CONNECT_ONLY, 2L);    
}