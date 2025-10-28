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
    // int num_threads = std::atoi(argv[1]);
    std::string host = argv[1];
    auto port = argv[2];
    auto message = argv[3];

    net::io_context ioc;
    tcp_stream sock(ioc);
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

    net::ip::tcp::resolver resolver(ioc);
    auto results = resolver.resolve(host, port);
    get_lowest_layer(wss).connect(results);

    wss.next_layer().handshake(net::ssl::stream_base::client);
    wss.handshake(host, "/ws");
    response_type response;
    wss.next_layer().handshake(net::ssl::stream_base::client);

    wss.write(net::buffer(std::string(message)));

    flat_buffer fb;

    wss.read(fb);

    wss.close(websocket::close_code::normal);

    return 0;
}