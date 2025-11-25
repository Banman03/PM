#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl.hpp>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <regex>

namespace net = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
using namespace boost::beast;
using namespace boost::beast::websocket;

// Simple JSON parsing to extract clobTokenIds from market data
std::vector<std::string> extract_clob_token_ids(const std::string& json_response) {
    std::vector<std::string> token_ids;
    std::regex clob_pattern(R"("clobTokenIds"\s*:\s*\[\s*"([^"]+)"(?:\s*,\s*"([^"]+)")?\s*\])");
    std::smatch matches;

    auto search_start = json_response.cbegin();
    while (std::regex_search(search_start, json_response.cend(), matches, clob_pattern)) {
        if (matches.size() > 1 && matches[1].matched) {
            token_ids.push_back(matches[1].str());
        }
        if (matches.size() > 2 && matches[2].matched) {
            token_ids.push_back(matches[2].str());
        }
        search_start = matches.suffix().first;
    }

    return token_ids;
}

// Query the markets API to get market information
std::string query_markets(const std::string& slug = "") {
    try {
        net::io_context ioc;
        net::ssl::context ctx(net::ssl::context::tlsv12_client);
        ctx.set_default_verify_paths();
        ctx.set_verify_mode(net::ssl::verify_peer);

        ssl_stream<tcp_stream> stream(ioc, ctx);

        std::string host = "gamma-api.polymarket.com";
        std::string port = "443";

        if (!SSL_set_tlsext_host_name(stream.native_handle(), host.c_str())) {
            throw beast::system_error(
                beast::error_code(
                    static_cast<int>(::ERR_get_error()),
                    net::error::get_ssl_category()),
                "Failed to set SNI hostname");
        }

        net::ip::tcp::resolver resolver(ioc);
        auto results = resolver.resolve(host, port);
        get_lowest_layer(stream).connect(results);
        stream.handshake(net::ssl::stream_base::client);

        std::string target = "/markets";
        if (!slug.empty()) {
            target += "?slug=" + slug;
        } else {
            target += "?limit=10&active=true";
        }

        http::request<http::string_body> req{http::verb::get, target, 11};
        req.set(http::field::host, host);
        req.set(http::field::user_agent, "Polymarket-CPP-Client/1.0");

        http::write(stream, req);

        beast::flat_buffer buffer;
        http::response<http::string_body> res;
        http::read(stream, buffer, res);

        beast::error_code ec;
        stream.shutdown(ec);

        return res.body();
    }
    catch (std::exception const& e) {
        std::cerr << "Error querying markets: " << e.what() << std::endl;
        return "";
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " --slug <market_slug>     # Query market by slug and connect to WebSocket\n"
                  << "  " << argv[0] << " --list                   # List 10 active markets\n"
                  << "  " << argv[0] << " <asset_id>               # Connect directly with asset ID\n"
                  << "\nExample:\n"
                  << "  " << argv[0] << " --slug will-trump-win-2024\n"
                  << "  " << argv[0] << " 21742633143463906290569050155826241533067272736897614950488156847949938836455\n";
        return 1;
    }

    std::string mode = argv[1];
    std::vector<std::string> asset_ids;

    // Handle different modes
    if (mode == "--list") {
        std::cout << "Querying active markets...\n";
        std::string response = query_markets();
        if (response.empty()) {
            std::cerr << "Failed to query markets\n";
            return 1;
        }

        auto token_ids = extract_clob_token_ids(response);
        std::cout << "\nFound " << token_ids.size() << " asset IDs in active markets:\n";
        for (const auto& id : token_ids) {
            std::cout << "  " << id << "\n";
        }
        return 0;
    }
    else if (mode == "--slug") {
        if (argc < 3) {
            std::cerr << "Error: --slug requires a market slug argument\n";
            return 1;
        }

        std::string slug = argv[2];
        std::cout << "Querying market: " << slug << "\n";
        std::string response = query_markets(slug);
        if (response.empty()) {
            std::cerr << "Failed to query market\n";
            return 1;
        }

        asset_ids = extract_clob_token_ids(response);
        if (asset_ids.empty()) {
            std::cerr << "No asset IDs found for market slug: " << slug << "\n";
            return 1;
        }

        std::cout << "Found " << asset_ids.size() << " asset ID(s):\n";
        for (const auto& id : asset_ids) {
            std::cout << "  " << id << "\n";
        }
    }
    else {
        // Direct asset ID mode
        asset_ids.push_back(argv[1]);
        std::cout << "Using provided asset ID: " << asset_ids[0] << "\n";
    }

    std::string host = "ws-subscriptions-clob.polymarket.com";
    std::string port = "443";

    try {
        net::io_context ioc;
        net::ssl::context ctx(net::ssl::context::tlsv12_client);
        ctx.set_default_verify_paths();
        ctx.set_verify_mode(net::ssl::verify_peer);

        websocket::stream<ssl_stream<tcp_stream>> wss(ioc, ctx);

        if (!SSL_set_tlsext_host_name(
                wss.next_layer().native_handle(),
                host.c_str()))
        {
            throw beast::system_error(
                beast::error_code(
                    static_cast<int>(::ERR_get_error()),
                    net::error::get_ssl_category()),
                "Failed to set SNI hostname");
        }

        std::cout << "Connecting to " << host << ":" << port << "...\n";
        net::ip::tcp::resolver resolver(ioc);
        auto results = resolver.resolve(host, port);
        get_lowest_layer(wss).connect(results);

        std::cout << "Performing SSL handshake...\n";
        wss.next_layer().handshake(net::ssl::stream_base::client);

        std::cout << "Performing WebSocket handshake...\n";
        wss.handshake(host, "/ws/market");

        // Build JSON subscription message for market channel
        std::ostringstream json;
        json << "{"
             << "\"type\":\"MARKET\","
             << "\"markets\":[],"
             << "\"assets_ids\":[";

        for (size_t i = 0; i < asset_ids.size(); ++i) {
            json << "\"" << asset_ids[i] << "\"";
            if (i < asset_ids.size() - 1) {
                json << ",";
            }
        }

        json << "]}";

        std::string subscription_msg = json.str();
        std::cout << "Subscribing to market channel with message: " << subscription_msg << "\n";
        wss.write(net::buffer(subscription_msg));

        std::cout << "Listening for messages (press Ctrl+C to stop)...\n\n";

        while (true) {
            flat_buffer buffer;
            wss.read(buffer);

            std::string message = beast::buffers_to_string(buffer.data());
            std::cout << "Received message:\n" << message << "\n\n";
        }

        wss.close(websocket::close_code::normal);
    }
    catch (std::exception const& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}