#include "websocket_client.hpp"
#include <iostream>
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/url.hpp>

namespace beast = boost::beast;
namespace websocket = beast::websocket;
namespace net = boost::asio;
namespace ssl = boost::asio::ssl;
using tcp = boost::asio::ip::tcp;

WebSocketClient::WebSocketClient() {
    // Configure SSL context to accept self-signed certificates
    ctx.set_verify_mode(ssl::verify_none);
    ctx.set_options(ssl::context::default_workarounds |
                   ssl::context::no_sslv2 |
                   ssl::context::no_sslv3 |
                   ssl::context::single_dh_use);
}

WebSocketClient::~WebSocketClient() {
    disconnect();
}

bool WebSocketClient::connect(const std::string& host, const std::string& port, const std::string& target) {
    try {
        // Resolve hostname
        auto const results = resolver.resolve(host, port);

        // Get the underlying socket
        auto& socket = beast::get_lowest_layer(ws);

        // Make the connection on the IP address we get from a lookup
        auto ep = net::connect(socket, results);

        // Update the host string for SNI
        std::string hostWithPort = host + ':' + std::to_string(ep.port());

        // Set SNI Hostname (many hosts need this to handshake successfully)
        if (!SSL_set_tlsext_host_name(ws.next_layer().native_handle(), host.c_str())) {
            beast::error_code ec{static_cast<int>(::ERR_get_error()), net::error::get_ssl_category()};
            throw beast::system_error{ec};
        }

        // Perform the SSL handshake
        ws.next_layer().handshake(ssl::stream_base::client);

        // Set a decorator to change the User-Agent of the handshake
        ws.set_option(websocket::stream_base::decorator(
            [](websocket::request_type& req) {
                req.set(beast::http::field::user_agent, "MultiFramework-Native-Client/1.0");
            }));

        // Perform the websocket handshake
        ws.handshake(hostWithPort, target);

        // Start the event loop in a separate thread
        shouldStop = false;
        ioThread = std::thread(&WebSocketClient::runEventLoop, this);

        if (onConnected) {
            onConnected();
        }

        return true;

    } catch (std::exception const& e) {
        std::cerr << "WebSocket connection error: " << e.what() << std::endl;
        return false;
    }
}

void WebSocketClient::disconnect() {
    shouldStop = true;

    if (ws.is_open()) {
        try {
            ws.close(websocket::close_code::normal);
        } catch (...) {
            // Ignore errors during close
        }
    }

    if (ioThread.joinable()) {
        ioThread.join();
    }

    if (onDisconnected) {
        onDisconnected();
    }
}

void WebSocketClient::send(const std::string& message) {
    if (!ws.is_open()) {
        std::cerr << "WebSocket not connected, cannot send message" << std::endl;
        return;
    }

    try {
        ws.write(net::buffer(message));
    } catch (std::exception const& e) {
        std::cerr << "WebSocket send error: " << e.what() << std::endl;
    }
}

void WebSocketClient::runEventLoop() {
    beast::flat_buffer buffer;

    while (!shouldStop && ws.is_open()) {
        try {
            // Read a message
            ws.read(buffer);

            // Convert to string
            std::string message = beast::buffers_to_string(buffer.data());
            buffer.clear();

            // Call message handler
            if (onMessage) {
                onMessage(message);
            }

        } catch (beast::system_error const& se) {
            if (se.code() != websocket::error::closed) {
                std::cerr << "WebSocket read error: " << se.code().message() << std::endl;
            }
            break;
        } catch (std::exception const& e) {
            std::cerr << "WebSocket event loop error: " << e.what() << std::endl;
            break;
        }
    }
}

