# importing the required packages and modules
from neo4j import GraphDatabase
import pandas as pd
import matplotlib.pyplot as plt
import powerlaw

uri = "bolt://127.0.0.1:7687"
username = "neo4j"
password = "Sunny1758#"

driver = GraphDatabase.driver(uri, auth=(username, password), encrypted=False)


try:
    # Create a session and run the desired query
    with driver.session() as session:
        # query to return all the nodes in thhe graph
        query1 = "MATCH (n) RETURN n"
        result = session.run(query1)
        for record in result:
            node = record["n"]
            print(node)

        # query to return the degree of every code in the graph
        # More the degree, more the co_authored relationships the specific node has
        query2 = (
            "MATCH (n) "
            "RETURN n, size([(n)--() | 1]) AS degree "
            "ORDER BY degree DESC"
        )
        result = session.run(query2)
        # Convert the result to pandas dataframe
        data = []
        for record in result:
            data.append({"Node": record["n"], "Degree": record["degree"]})

        df = pd.DataFrame(data)
        print(df)

        # Calculate the average degree using the DataFrame
        avg_degree = df["Degree"].mean()
        print("The average degree of the graph is:", avg_degree)

        # query to make degree array to calculate powerlaw distribution
        degrees = []
        query3 = (
            "MATCH (n) " 
            "RETURN n, size([(n)--() | 1]) AS degree "
        )
        result = session.run(query3)
        for record in result:
            degrees.append(record["degree"])

        # Analyze the distribution using powerlaw library
        fit = powerlaw.Fit(degrees)

        # Plot the data and distribution
        fit.plot_pdf(color='b', linewidth=2)
        fit.power_law.plot_pdf(color='r', linestyle='--', ax=plt.gca())
        plt.xlabel("Node Degree")
        plt.ylabel("Probability Density")
        plt.title("Node Degree Distribution")
        plt.show()

        print("Power-law alpha:", fit.power_law.alpha)

        # query to find the density of the graph
        query4 = (
            "MATCH () - [e] - ()"
            "WITH COUNT(DISTINCT e) AS edgeCount "
            "MATCH (n) "
            "WITH edgeCount, COUNT(n) AS nodeCount "
            "RETURN edgeCount / nodeCount AS density "
        )
        result = session.run(query4)
        density = result.single()["density"]
        print("The density of the graph is:", density)

        # query to find if the given graph has self loops or not
        query5 = (
            "MATCH (n) - [e] - (n) "
            "RETURN COUNT(e) AS selfLoopCount"
        )
        result = session.run(query5)
        selfcount = result.single()["selfLoopCount"]
        print("The self loops in the graph are:", selfcount)

except Exception as e:
    print("Error:", e)
finally:
    # Close the driver when you're done
    driver.close()
