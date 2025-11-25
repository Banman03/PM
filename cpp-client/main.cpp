#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl.hpp>
#include <stdlib.h>
#include <string>
#include <iostream>


namespace net = boost::asio;
namespace beast = boost::beast;
using namespace boost::beast;
using namespace boost::beast::websocket;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <asset_id>\n"
                  << "Example: " << argv[0] << " 21742633143463906290569050155826241533067272736897614950488156847949938836455\n";
        return 1;
    }

    std::string asset_id = argv[1];
    std::string host = /*"echo.websocket.org";*/"ws-subscriptions-clob.polymarket.com";
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
        wss.handshake(host, "/wss/");

        std::string subscription_msg = "market " + asset_id;
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