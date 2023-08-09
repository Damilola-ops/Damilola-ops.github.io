import { FaGithub, FaLinkedin, FaTwitter, FaCopyright } from "react-icons/fa";
import "./styles/Message.css";
function Message() {
  return (
    <>
      <body>
        <nav>
          <div className="leftClass">Home</div>
          <div className="rightClass">
            <div>categories</div>
            <div>tags</div>
          </div>
        </nav>
        <main>
          <div className="main-container">
            <div className="text">
              <div>
                <h1>Damilola John Oduguwa</h1>
              </div>
              <div className="skills">
                Software engineer specializing in machine learning engineering
                and NLP, living and studying at the University of Lagos
              </div>
            </div>
            <div className="icon-link">
              <a
                href=" https://twitter.com/damilojohn "
                target="_blank"
                rel="noopener noreferrer"
                title="twitter"
              >
                <i className="FaTwitter">
                  <FaTwitter />
                </i>
              </a>
              <a
                href=" https://github.com/damilojohn "
                target="_blank"
                rel="noopener noreferrer"
                title="github"
              >
                <i className="FaGithub">
                  <FaGithub />
                </i>
              </a>
              <a
                href=" https://www.linkedin.com/in/oduguwa-damilola-b089131a8/ "
                target="_blank"
                rel="noopener noreferrer"
                title="linkedin"
              >
                <i className="FaLinkedin">
                  <FaLinkedin />
                </i>
              </a>
            </div>
            <div id="project-link">
              <button id="articles">
                <a
                  href="https://medium.com/@oduguwadamilola40"
                  target="_blank"
                  rel="noopener"
                >
                  Articles
                </a>
              </button>
              <button id="project">Projects</button>
              <button id="cv">
                <a
                  href="http://127.0.0.1:5501/docs/Oduguwa%20Damilola%20John%20Resume%20(1)%20(1).pdf"
                  target="_blank"
                  rel="noopener"
                >
                  CV
                </a>
              </button>
            </div>
          </div>
        </main>
        <footer>
          <div className="footer">
            <i className="FaCopyright">
              {" "}
              <FaCopyright></FaCopyright>
            </i>{" "}
            2023 Damilola Makinde
          </div>
        </footer>
      </body>
    </>
  );
}
export default Message;
