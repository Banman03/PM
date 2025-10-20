#include <string>;
#include <curl/curl.h>;
#include "constants.hpp";

void curl_ws_init(CURL* handler, CURLcode* response) {
    std::string ws_url = constants::WEBSOCKET_URL;

}