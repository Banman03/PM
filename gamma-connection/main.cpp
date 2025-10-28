// #include <boost/thread.hpp>;
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <stdlib.h>
#include <string>
// #include <vector>
#include <iostream>

#include <boost/beast.hpp>
#include <boost/asio.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket/ssl.hpp>

namespace net = boost::asio;
namespace beast = boost::beast;
namespace ssl = net::ssl;
namespace ip = net::ip;
using namespace boost::beast;
using namespace boost::beast::websocket;

int main(int argc, char** argv) {
    // int num_threads = std::atoi(argv[1]);
    std::string host = argv[1];
    auto port = argv[2];
    auto message = argv[3];

    net::io_context ioc;
    tcp_stream sock(ioc);
    net::ssl::context ctx(net::ssl::context::tlsv12);
    stream<ssl_stream<tcp_stream>> wss(net::make_strand(ioc), ctx);

    net::ip::tcp::resolver resolver(ioc);
    auto results = resolver.resolve(host, port);
    get_lowest_layer(wss).connect(results);

    std::cout << "here3" << std::endl;
    response_type response;

    wss.handshake(response, host, "/");

    net::mutable_buffer b(message, sizeof(message));
    wss.write(b);

    flat_buffer fb;
    
    wss.read(fb);

    std::string s(net::buffers_begin(fb.data()), net::buffers_end(fb.data()));
    std::cout
        << "reading: " << s << std::endl;

    wss.close(beast::websocket::close_code::normal);

    
    return 0;
}