#pragma once

#include <string>
#include <memory>
#include <sstream>

#include <Poco/JSON/Object.h>
#include <Poco/JSON/Parser.h>
#include <Poco/Dynamic/Var.h>
#include <Poco/Net/Context.h>
#include <Poco/Net/HTTPRequest.h>
#include <Poco/Net/HTTPResponse.h>
#include <Poco/Net/HTTPSClientSession.h>
#include <Poco/Net/HTTPClientSession.h>
#include <Poco/Net/NetException.h>
#include <Poco/URI.h>
#include <Poco/Timespan.h>
#include "nano_graphrag/utils/Log.hpp"

namespace nano_graphrag
{

class RestClient
{
public:
  RestClient() = default;
  ~RestClient() = default;

  void set_uri(const std::string& uri)
  {
    uri_ = uri;
  }
  void set_method(const std::string& method)
  {
    method_ = method;
  }
  void set_ssl_verify(bool v)
  {
    ssl_verify_ = v;
  }
  void set_auth_bearer(const std::string& token)
  {
    auth_type_ = "Bearer";
    api_key_ = token;
  }
  void set_auth_type(const std::string& t)
  {
    auth_type_ = t;
  }

  Poco::JSON::Object::Ptr post_json(Poco::JSON::Object& body_json, const std::string& uri)
  {
    Poco::URI uri_obj(uri);

    debug_log("[RestClient] POST ", uri);

    std::ostringstream body_stream;
    body_json.stringify(body_stream);

    Poco::Net::HTTPRequest request(method_, uri_obj.getPathEtc());
    request.setVersion(Poco::Net::HTTPMessage::HTTP_1_1);
    request.setHost(uri_obj.getHost());
    request.setContentType("application/json");
    request.setContentLength(body_stream.str().size());
    request.set("Accept", "application/json");

    if (auth_type_ == "Bearer" && !api_key_.empty())
    {
      request.setCredentials(auth_type_, api_key_);
    }

    std::unique_ptr<Poco::Net::HTTPClientSession> session_ptr;
    if (uri_obj.getScheme() == "https")
    {
      Poco::Net::Context::Params params;
      params.verificationMode =
          ssl_verify_ ? Poco::Net::Context::VERIFY_STRICT : Poco::Net::Context::VERIFY_NONE;
      if (ssl_verify_)
        params.caLocation = "/etc/ssl/certs";
      Poco::Net::Context::Ptr context = new Poco::Net::Context(Poco::Net::Context::CLIENT_USE, params);
      session_ptr =
          std::make_unique<Poco::Net::HTTPSClientSession>(uri_obj.getHost(), uri_obj.getPort(), context);
    }
    else
    {
      session_ptr = std::make_unique<Poco::Net::HTTPClientSession>(uri_obj.getHost(), uri_obj.getPort());
    }

    try
    {
      session_ptr->setTimeout(Poco::Timespan(60, 0));
      std::ostream& os = session_ptr->sendRequest(request);
      body_json.stringify(os);
      debug_log("[RestClient] request sent, awaiting response...");
    }
    catch (const Poco::Net::NetException& e)
    {
      debug_log("[RestClient] exception during sendRequest: ", e.displayText());
      throw;
    }

    Poco::Net::HTTPResponse response;
    std::istream& rs = session_ptr->receiveResponse(response);

    debug_log("[RestClient] response status=", static_cast<int>(response.getStatus()), " ",
              response.getReason());

    if (response.getStatus() != Poco::Net::HTTPResponse::HTTP_OK)
    {
      throw Poco::Net::NetException("HTTP Error: " + std::to_string(response.getStatus()) + " " +
                                    response.getReason());
    }

    if (response.getContentType() == "text/event-stream" || response.getContentType() == "application/"
                                                                                         "x-ndjson")
    {
      debug_log("[RestClient] stream content-type not supported");
      throw Poco::Net::NetException("HTTP stream not supported");
    }

    if (response.getChunkedTransferEncoding())
    {
      debug_log("[RestClient] chunked transfer not supported");
      throw Poco::Net::NetException("HTTP Chunked Transfer Encoding not supported");
    }

    Poco::JSON::Parser parser;
    Poco::Dynamic::Var result = parser.parse(rs);
    return result.extract<Poco::JSON::Object::Ptr>();
  }

private:
  std::string uri_;
  std::string method_{ "POST" };
  bool ssl_verify_{ true };
  std::string auth_type_{ "Bearer" };
  std::string api_key_;
};

}  // namespace nano_graphrag
