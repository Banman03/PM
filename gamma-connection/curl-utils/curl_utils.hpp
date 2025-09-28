#pragma once

#include <curl/curl.h>

void curl_init(CURL *curl_handler_main, CURLcode *curl_setup_response_main);
void curl_destroy(CURL *curl_handler_main);