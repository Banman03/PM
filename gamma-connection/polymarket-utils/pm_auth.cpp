#include <curl/curl.h>;
#include <stdlib.h>;
#include <string>;
#include <boost/uuid.hpp>;
#include <vector>;
#include <stdlib.h>;
#include <boost/process/environment.hpp>;
#include <boost/process/env.hpp>;

struct api_key_auth {
    boost::uuids::uuid key;
    std::string secret;
    std::string passphrase;
};

struct type_element {
    std::string name;
    std::string type;
};

struct eip_712 {
    struct domain {
        std::string name;
        std::string version;
        int chainId;
    };

    struct types {
        std::vector<type_element> clobAuth;
    };

    struct value {
        std::string address;
        long long timestamp;
        int nonce;
        std::string message;
    };

    domain d;
    types t;
    value v;

    eip_712(const std::string &name,
            const std::string &version,
            int chainId,
            const std::vector<type_element> &clobAuth,
            const std::string &address,
            long long timestamp,
            int nonce,
            const std::string &message)
        : d{name, version, chainId},
          t{clobAuth},
          v{address, timestamp, nonce, message} {}
};

void polymarket_api_key_auth_init(CURL* handler) {
    boost::process::basic_environment env = boost::process::env();
    
    std::vector<type_element> clobAuthTypes = {
        {"address", "address"},
        {"timestamp", "string"},
        {"nonce", "uint256"},
        {"message", "string"}};

    eip_712 eip_message(
        ,
        ,
        137,
        clobAuthTypes,
        
    )
}

void polymarket_private_key_auth_init(CURL* handler) {
    
}